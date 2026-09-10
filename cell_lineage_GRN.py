from typing import Optional, Union
from torch_geometric.typing import Adj

import numpy as np
import pandas as pd
import networkx as nx
from tqdm import trange
# from tqdm.auto import tqdm
import math
from scanpy import AnnData

import torch
from torch import Tensor, nn
import torch.nn.functional as F

import torch_geometric as pyg
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn import Linear, BatchNorm, DeepGraphInfomax
from torch_geometric.utils import (
    remove_self_loops,
    add_self_loops,
    softmax,
    to_undirected,
)
from torch_geometric.data import Data

from TEDC2L_result_object import TEDC2LResults


class GraphAttention_layer(MessagePassing):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 attention_type: str = 'COS',
                 flow: str = 'source_to_target',
                 heads: int = 1,
                 concat: bool = True,
                 dropout: float = 0.0,
                 add_self_loops: bool = True,
                 to_undirected: bool = False,
                 edge_gate_beta: float = 1.0,  # NEW: Edge-level gating parameter
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        kwargs.setdefault('flow', flow)
        super(GraphAttention_layer, self).__init__(node_dim=0, **kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.attention_type = attention_type
        self.to_undirected = to_undirected
        self.edge_gate_beta = edge_gate_beta  # NEW

        # AD: Additive(GAT); SD: abs(scaled-dot product); COS: abs(cosine)
        assert attention_type in ['SD', 'COS', 'AD']
        self.lin_l = Linear(input_dim, heads * output_dim, bias=False,
                            weight_initializer='glorot')
        self.lin_r = Linear(input_dim, heads * output_dim, bias=False,
                            weight_initializer='glorot')

        if self.attention_type == 'AD':
            self.att_l = nn.Parameter(Tensor(1, heads, output_dim))
            self.att_r = nn.Parameter(Tensor(1, heads, output_dim))
        else:
            self.register_parameter('att_l', None)
            self.register_parameter('att_r', None)

        if concat:
            self.bias = nn.Parameter(Tensor(heads * output_dim))
            self.weight_concat = nn.Parameter(Tensor(heads * output_dim, output_dim))
        else:
            self.bias = nn.Parameter(Tensor(output_dim))
            self.register_parameter('weight_concat', None)

        self._alpha = None
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        glorot(self.att_l)
        glorot(self.att_r)
        zeros(self.bias)
        glorot(self.weight_concat)

    def forward(self, x: Tensor, edge_index: Adj, x_auxiliary: Tensor,
                edge_aux_text: Optional[Tensor] = None,  # NEW: Edge auxiliary text features
                return_attention_weights: Optional[bool] = None):
        N, H, C = x.size(0), self.heads, self.output_dim

        if self.to_undirected:
            edge_index = to_undirected(edge_index)

        # Handle self-loops and edge auxiliary features carefully
        if self.add_self_loops:
            edge_index, edge_aux_text = remove_self_loops(edge_index, edge_attr=edge_aux_text)
            edge_index, edge_aux_text = add_self_loops(edge_index, edge_attr=edge_aux_text,
                                                       num_nodes=N, fill_value=0.0)
        else:
            edge_index, edge_aux_text = remove_self_loops(edge_index, edge_attr=edge_aux_text)

        x_l = self.lin_l(x).view(-1, H, C)
        x_r = self.lin_r(x).view(-1, H, C)

        if self.attention_type == 'AD':
            out = self.propagate(edge_index, x=(x_l, x_r), x_norm=None,
                                 x_auxiliary=x_auxiliary, edge_aux_text=edge_aux_text, size=None)
        elif self.attention_type == 'COS':
            x_norm_l = F.normalize(x_l, p=2., dim=-1)
            x_norm_r = F.normalize(x_r, p=2., dim=-1)
            out = self.propagate(edge_index, x=(x_l, x_r), x_norm=(x_norm_l, x_norm_r),
                                 x_auxiliary=x_auxiliary, edge_aux_text=edge_aux_text, size=None)
        else:  # SD
            out = self.propagate(edge_index, x=(x_l, x_r), x_norm=None,
                                 x_auxiliary=x_auxiliary, edge_aux_text=edge_aux_text, size=None)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.output_dim)
            out += self.bias
            out = torch.matmul(out, self.weight_concat)
        else:
            out = out.mean(dim=1)
            out += self.bias

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            return out, (edge_index, alpha)
        else:
            return out

    def message(self, edge_index_i: Tensor, x_i: Tensor, x_j: Tensor,
                x_norm_i: Optional[Tensor], x_norm_j: Optional[Tensor],
                x_auxiliary_j: Tensor,
                edge_aux_text: Optional[Tensor],  # NEW: Edge auxiliary text features
                size_i: Optional[int]):

        Tau = 1.0  # temperature hyperparameter

        # Compute base attention scores (original logic)
        if self.attention_type == 'AD':
            alpha = (x_j * self.att_l).sum(-1) + (x_i * self.att_r).sum(-1)
            alpha = F.leaky_relu(alpha, 0.2)
        elif self.attention_type == 'COS':
            alpha = torch.abs((x_norm_i * x_norm_j).sum(dim=-1))
            Tau = 0.25
        else:  # 'SD'
            alpha = torch.abs((x_i * x_j).sum(dim=-1)) / math.sqrt(self.output_dim)

        # Apply edge-level gating (NEW with proper tensor handling)
        # Compute gate values: g_ij = β * x_auxiliary_j + (1-β) * edge_aux_text
        gate = self.edge_gate_beta * x_auxiliary_j  # Node-level auxiliary (original)

        if edge_aux_text is not None and self.edge_gate_beta < 1.0:
            # Ensure edge_aux_text has the correct shape and size
            if edge_aux_text.size(0) != alpha.size(0):
                # Handle size mismatch by padding or truncating
                if edge_aux_text.size(0) < alpha.size(0):
                    # Pad with zeros if edge_aux_text is smaller
                    padding_size = alpha.size(0) - edge_aux_text.size(0)
                    padding = torch.zeros(padding_size, edge_aux_text.size(1),
                                          device=edge_aux_text.device, dtype=edge_aux_text.dtype)
                    edge_aux_text = torch.cat([edge_aux_text, padding], dim=0)
                else:
                    # Truncate if edge_aux_text is larger
                    edge_aux_text = edge_aux_text[:alpha.size(0)]

            # Ensure proper dimension matching
            if edge_aux_text.dim() == 2 and edge_aux_text.size(1) == 1:
                # Expand to match head dimension
                edge_aux_text_expanded = edge_aux_text.expand(-1, alpha.size(1))
            elif edge_aux_text.dim() == 1:
                # Add head dimension and expand
                edge_aux_text_expanded = edge_aux_text.unsqueeze(1).expand(-1, alpha.size(1))
            else:
                edge_aux_text_expanded = edge_aux_text

            # Apply ReLU for non-negativity and combine with node auxiliary
            text_gate_component = (1.0 - self.edge_gate_beta) * torch.relu(edge_aux_text_expanded)

            # Ensure size match before addition
            if text_gate_component.size(0) == gate.size(0):
                gate = gate + text_gate_component
            else:
                # Fallback: only use node auxiliary if sizes don't match
                pass

        # Apply gating to attention scores
        alpha = gate * alpha

        # Apply softmax with temperature (original logic)
        alpha = softmax(alpha / Tau, edge_index_i, num_nodes=size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.view(-1, self.heads, 1)

    def __repr__(self):
        return '{}({}, {}, heads={}, type={}, edge_gate_beta={})'.format(
            self.__class__.__name__,
            self.input_dim,
            self.output_dim,
            self.heads,
            self.attention_type,
            self.edge_gate_beta)


class GRN_Encoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 heads_num: int = 1,
                 dropout: float = 0.0,
                 attention_type: str = 'COS',
                 edge_gate_beta: float = 1.0):  # NEW: Edge-level gating parameter
        super(GRN_Encoder, self).__init__()
        assert attention_type in ['SD', 'COS', 'AD']

        self.att_weights_first = None
        self.att_weights_second = None
        self.x_embs = None
        self.edge_gate_beta = edge_gate_beta  # NEW

        self.x_input = nn.Linear(input_dim, hidden_dim)
        self.c = nn.Parameter(Tensor(1))
        self.d = nn.Parameter(Tensor(1))
        self.c.data.fill_(1.6)
        self.d.data.fill_(torch.log(torch.tensor(19.0)))

        self.act = nn.GELU()
        self.layers = nn.ModuleList([])
        dims = [hidden_dim, hidden_dim]  # 2 layers are used
        for l in range(len(dims)):
            concat = True  # if l==0 else False
            last_dim = hidden_dim if l < len(dims) - 1 else output_dim
            self.layers.append(nn.ModuleList([
                BatchNorm(dims[l]),
                # in-coming
                GraphAttention_layer(dims[l], dims[l], heads=heads_num,
                                     concat=concat, dropout=dropout,
                                     attention_type=attention_type,
                                     edge_gate_beta=edge_gate_beta),  # NEW: Pass edge_gate_beta
                # out-going
                GraphAttention_layer(dims[l], dims[l], heads=heads_num,
                                     concat=concat, dropout=dropout,
                                     attention_type=attention_type,
                                     flow='target_to_source',
                                     edge_gate_beta=edge_gate_beta),  # NEW: Pass edge_gate_beta
                nn.Sequential(
                    nn.Linear(dims[l] * 2, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, last_dim),
                ),
            ]))
        self.project = nn.Linear(output_dim, output_dim * 4)

    def forward(self, data: dict):
        x, edge_index = data['x'], data['edge_index']

        # Node auxiliary features (original logic)
        if 'node_score_auxiliary' in data:
            x_auxiliary = torch.sigmoid(self.c * data['node_score_auxiliary'] - self.d)
        else:
            x_auxiliary = torch.ones(x.size(0), dtype=torch.float32, device=x.device).view(-1, 1)

        # Edge auxiliary features (NEW)
        edge_aux_text = data.get('edge_aux_text', None)

        x = self.x_input(x)

        att_weights_in, att_weights_out = [], []
        for norm, attn_in, attn_out, ffn in self.layers:
            x = norm(x)

            # Pass edge auxiliary features to attention layers
            x_in, att_weights_in_ = attn_in(x, edge_index, x_auxiliary,
                                            edge_aux_text=edge_aux_text,
                                            return_attention_weights=True)
            x_out, att_weights_out_ = attn_out(x, edge_index, x_auxiliary,
                                               edge_aux_text=edge_aux_text,
                                               return_attention_weights=True)
            x = ffn(torch.cat((self.act(x_in), self.act(x_out)), 1))

            att_weights_in.append(att_weights_in_)
            att_weights_out.append(att_weights_out_)

        self.x_embs = x
        # [edge_index, att_in, att_out]
        self.att_weights_first = (att_weights_in[0][0], att_weights_in[0][1], att_weights_out[0][1])
        self.att_weights_second = (att_weights_in[1][0], att_weights_in[1][1], att_weights_out[1][1])

        return self.project(x)


class NetModel(object):
    """
    The model for constructing the cell-lineage-specific GRN with text similarity enhancements.

    Args:
        hidden_dim (int, optional): hidden dimension of the GNN encoder (default: 128)
        output_dim (int, optional): output dimension of the GNN encoder (default: 64)
        heads (int, optional): number of heads for the multi-head attention (default: 4)
        attention_type (str, optional): type of attention scoring function ('COS', 'SD', 'AD') (default: 'COS')
        dropout (float, optional): dropout rate (default: 0.1)
        miu (float, optional): parameter (0~1) for considering the importance of attention coefficients of the first GNN layer (default: 0.5)
        epochs (int, optional): number of epochs for one repeat (default: 350)
        repeats (int, optional): number of repeats (default: 5)
        seed (int, optional): random seed. Set to -1 means no random seed is assigned (default: -1)
        cuda (int, optional): an integer greater than -1 indicates the GPU device number and -1 indicates the CPU device
        edge_gate_beta (float, optional): edge-level gating parameter (1.0=node-level only, 0.0=text-similarity only) (default: 1.0)
        loss_text_align (float, optional): weight for graph-text contrastive learning loss (0.0=disabled) (default: 0.0)
        contrastive_proj_dim (int, optional): projection dimension for contrastive learning (default: 128)
        contrastive_temperature (float, optional): temperature parameter for contrastive learning (default: 0.07)
    """

    def __init__(self,
                 hidden_dim: int = 128,
                 output_dim: int = 64,
                 heads: int = 4,
                 attention_type: str = 'COS',
                 dropout: float = 0.1,
                 miu: float = 0.5,
                 epochs: int = 350,
                 repeats: int = 5,
                 seed: int = -1,
                 cuda: int = -1,
                 # New text enhancement parameters
                 edge_gate_beta: float = 1.0,
                 loss_text_align: float = 0.0,
                 contrastive_proj_dim: int = 128,
                 contrastive_temperature: float = 0.07,
                 ):
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.heads = heads
        self.attention_type = attention_type
        self.dropout = dropout
        self.miu = miu
        self.epochs = epochs
        self.repeats = repeats

        # Text enhancement parameters
        self.edge_gate_beta = edge_gate_beta
        self.loss_text_align = loss_text_align
        self.contrastive_proj_dim = contrastive_proj_dim
        self.contrastive_temperature = contrastive_temperature

        if seed > -1:
            pyg.seed_everything(seed)
            torch.backends.cudnn.deterministic = True
        self.cuda = cuda

        self._idx_GeneName_map = None
        self._att_coefs = None
        self._node_embs = None
        self._adata = None

        self.GRN_predicted = None

        # Text enhancement attributes
        self._text_embeddings = None
        self._contrastive_losses = []

    def __get_PYG_data(self, adata: AnnData) -> Data:
        # edge index
        source_nodes = adata.uns['edgelist']['from'].tolist()
        target_nodes = adata.uns['edgelist']['to'].tolist()
        edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)

        # node feature
        x = torch.from_numpy(adata.to_df().T.to_numpy())     #原代码
        # x = torch.from_numpy(adata.to_df().to_numpy())
        pyg_data = Data(x=x, edge_index=edge_index)

        # node auxiliary score - Differential expression: logFC
        if 'node_score_auxiliary' in adata.var:
            pyg_data.node_score_auxiliary = torch.tensor(adata.var['node_score_auxiliary'].to_numpy(),
                                                         dtype=torch.float32).view(-1, 1)
        else:
            print('  Warning: Auxiliary gene scores (e.g., differential expression level) are not considered!')

        # Edge auxiliary information for text similarity (NEW)
        if 'gene_text_emb' in adata.varm:
            print('  Adding edge-level text similarity auxiliary information...')
            gene_text_emb = torch.tensor(adata.varm['gene_text_emb'], dtype=torch.float32)

            # Check if embeddings are non-zero
            if torch.any(gene_text_emb != 0):
                # Normalize embeddings for cosine similarity
                gene_text_emb_norm = torch.nn.functional.normalize(gene_text_emb, p=2, dim=1)

                # Compute cosine similarity for each edge
                src_indices = edge_index[0]  # source nodes
                dst_indices = edge_index[1]  # target nodes

                src_embs = gene_text_emb_norm[src_indices]  # [num_edges, emb_dim]
                dst_embs = gene_text_emb_norm[dst_indices]  # [num_edges, emb_dim]

                # Cosine similarity for each edge
                edge_text_sim = (src_embs * dst_embs).sum(dim=1, keepdim=True)  # [num_edges, 1]

                # Apply ReLU to ensure non-negative values for gate
                edge_text_sim = torch.clamp(edge_text_sim, min=0.0)

                pyg_data.edge_aux_text = edge_text_sim

                print(f'    Edge text similarity range: [{edge_text_sim.min():.4f}, {edge_text_sim.max():.4f}]')
                print(f'    Mean edge text similarity: {edge_text_sim.mean():.4f}')
            else:
                # All embeddings are zero, create zero auxiliary features
                pyg_data.edge_aux_text = torch.zeros(edge_index.size(1), 1, dtype=torch.float32)
                print('    Text embeddings are all zero, using zero edge auxiliary features')
        else:
            # No text embeddings available, don't create edge_aux_text
            # This maintains backward compatibility
            pass

        self._idx_GeneName_map = adata.varm['idx_GeneName_map']
        self._adata = adata

        return pyg_data

    @staticmethod
    def __corruption(data: Data) -> Data:
        x, edge_index = data['x'], data['edge_index']
        data_neg = Data(x=x[torch.randperm(x.size(0))], edge_index=edge_index)
        if 'node_score_auxiliary' in data:
            node_score_auxiliary = data['node_score_auxiliary']
            data_neg.node_score_auxiliary = node_score_auxiliary[torch.randperm(node_score_auxiliary.size(0))]
        return data_neg

    @staticmethod
    def __summary(z, *args, **kwargs) -> torch.Tensor:
        # return torch.sigmoid(z.mean(dim=0))
        return torch.sigmoid(torch.cat((3 * z.mean(dim=0).unsqueeze(0),
                                        z.max(dim=0)[0].unsqueeze(0),
                                        z.min(dim=0)[0].unsqueeze(0),
                                        2 * z.median(dim=0)[0].unsqueeze(0),
                                        ), dim=0))

    @staticmethod
    def __train(data, model, optimizer):
        model.train()
        optimizer.zero_grad()
        pos_z, neg_z, summary = model(data)
        loss = model.loss(pos_z, neg_z, summary)
        loss.backward()
        optimizer.step()
        return float(loss.item())

    @staticmethod
    def __get_encoder_results(data, model):
        model.eval()
        emb_last = model(data)
        return model.x_embs, model.att_weights_first, model.att_weights_second, emb_last

    def run(self, adata: AnnData, showProgressBar: bool = True):

        print('[1] - Constructing cell-lineage-specific GRN...')
        print('  Lineage - {}: '.format(adata.uns['name']))

        # Log text enhancement settings
        if self.edge_gate_beta < 1.0:
            print(f'  Using edge-level gating (β={self.edge_gate_beta})')
        if self.loss_text_align > 0:
            print(f'  Using graph-text contrastive learning (λ={self.loss_text_align})')

        if self.cuda == -1:
            device = "cpu"
        else:
            device = 'cuda:%s' % self.cuda

        ## Data for pyg input
        data = self.__get_PYG_data(adata).to(device)
        input_dim = data.num_node_features

        # Initialize contrastive learning components if needed
        contrastive_proj_gnn = None
        contrastive_proj_text = None
        text_embeddings = None

        if self.loss_text_align > 0 and hasattr(data, 'edge_aux_text'):
            # Get text embedding dimension from adata
            if 'gene_text_emb' in adata.varm:
                text_dim = adata.varm['gene_text_emb'].shape[1]
                text_embeddings = torch.tensor(adata.varm['gene_text_emb'],
                                               dtype=torch.float32, device=device)

                # Create projection heads
                contrastive_proj_gnn = nn.Sequential(
                    nn.Linear(self.output_dim, self.contrastive_proj_dim), # <--- 更改为这一行
                    nn.ReLU(),
                    nn.Linear(self.contrastive_proj_dim, self.contrastive_proj_dim)
                ).to(device)

                contrastive_proj_text = nn.Linear(text_dim, self.contrastive_proj_dim, bias=False).to(device)

                print(f'  Initialized contrastive learning: text_dim={text_dim}, proj_dim={self.contrastive_proj_dim}')

        ## Run for many times and take the average
        att_weights_all = []
        emb_out_avg = 0
        self._contrastive_losses = []

        for rep in range(self.repeats):
            ## Encoder & Model & Optimizer
            encoder = GRN_Encoder(input_dim, self.hidden_dim, self.output_dim, self.heads,
                                  dropout=self.dropout, attention_type=self.attention_type,
                                  edge_gate_beta=self.edge_gate_beta).to(device)
            DGI_model = DeepGraphInfomax(hidden_channels=self.output_dim * 4,
                                         encoder=encoder,
                                         summary=self.__summary,
                                         corruption=self.__corruption).to(device)

            # Setup optimizer including contrastive learning parameters
            optimizer_params = list(DGI_model.parameters())
            if contrastive_proj_gnn is not None:
                optimizer_params += list(contrastive_proj_gnn.parameters())
                optimizer_params += list(contrastive_proj_text.parameters())

            optimizer = torch.optim.Adam(optimizer_params, lr=1e-4, weight_decay=5e-4)

            ## Train
            best_encoder = encoder.state_dict()
            best_contrastive_gnn = contrastive_proj_gnn.state_dict() if contrastive_proj_gnn is not None else None
            best_contrastive_text = contrastive_proj_text.state_dict() if contrastive_proj_text is not None else None

            min_loss = np.inf
            rep_contrastive_losses = []

            if showProgressBar:
                with trange(self.epochs, ncols=100) as t:
                    for epoch in t:
                        # Enhanced training step with contrastive learning
                        loss, contrastive_loss = self.__enhanced_train(
                            data, DGI_model, optimizer, encoder,
                            contrastive_proj_gnn, contrastive_proj_text, text_embeddings
                        )

                        rep_contrastive_losses.append(contrastive_loss)
                        t.set_description('  Iter: {}/{}'.format(rep + 1, self.repeats))

                        if epoch < self.epochs - 1:
                            if contrastive_loss > 0:
                                t.set_postfix(dgi_loss=f'{loss - self.loss_text_align * contrastive_loss:.4f}',
                                              text_loss=f'{contrastive_loss:.4f}')
                            else:
                                t.set_postfix(loss=f'{loss:.4f}')

                        if min_loss > loss:
                            min_loss = loss
                            best_encoder = encoder.state_dict()
                            if contrastive_proj_gnn is not None:
                                best_contrastive_gnn = contrastive_proj_gnn.state_dict()
                                best_contrastive_text = contrastive_proj_text.state_dict()

                        if epoch == self.epochs - 1:
                            if contrastive_loss > 0:
                                t.set_postfix(dgi_loss=f'{loss - self.loss_text_align * contrastive_loss:.4f}',
                                              text_loss=f'{contrastive_loss:.4f}',
                                              min_total=f'{min_loss:.4f}')
                            else:
                                t.set_postfix(loss=f'{loss:.4f}', min_loss=f'{min_loss:.4f}')
            else:
                print('  Iter: {}/{}'.format(rep + 1, self.repeats), end='... ')
                for epoch in range(self.epochs):
                    loss, contrastive_loss = self.__enhanced_train(
                        data, DGI_model, optimizer, encoder,
                        contrastive_proj_gnn, contrastive_proj_text, text_embeddings
                    )
                    rep_contrastive_losses.append(contrastive_loss)

                    if min_loss > loss:
                        min_loss = loss
                        best_encoder = encoder.state_dict()
                        if contrastive_proj_gnn is not None:
                            best_contrastive_gnn = contrastive_proj_gnn.state_dict()
                            best_contrastive_text = contrastive_proj_text.state_dict()

                if contrastive_loss > 0:
                    print('Min_total_loss: {:.4f} (DGI: {:.4f}, Text: {:.4f})'.format(
                        min_loss, min_loss - self.loss_text_align * contrastive_loss, contrastive_loss))
                else:
                    print('Min_train_loss: {:.4f}'.format(min_loss))

            ## Get the result of the best model
            encoder.load_state_dict(best_encoder)
            if contrastive_proj_gnn is not None:
                contrastive_proj_gnn.load_state_dict(best_contrastive_gnn)
                contrastive_proj_text.load_state_dict(best_contrastive_text)

            gene_emb, weights_first, weights_second, emb_last = self.__get_encoder_results(data, encoder)
            gene_emb = gene_emb.cpu().detach().numpy()

            # Use the average of multi-head's attention weights
            weights_first = (weights_first[0].cpu().detach(),
                             torch.cat((weights_first[1].mean(dim=1, keepdim=True),
                                        weights_first[2].mean(dim=1, keepdim=True)), 1).cpu().detach())
            weights_second = (weights_second[0].cpu().detach(),
                              torch.cat((weights_second[1].mean(dim=1, keepdim=True),
                                         weights_second[2].mean(dim=1, keepdim=True)), 1).cpu().detach())

            # Combine the attention coefficients of the first and the second layer
            att_weights = self.miu * weights_first[1] + (1 - self.miu) * weights_second[1]
            att_weights_all.append(att_weights)
            emb_out_avg = emb_out_avg + gene_emb
            self._contrastive_losses.extend(rep_contrastive_losses)

            if device != 'cpu':
                torch.cuda.empty_cache()

        ## Take the average of multiple runs
        if self.repeats > 1:
            att_weights_all = torch.stack((att_weights_all), 0)
        else:
            att_weights_all = att_weights_all[0].unsqueeze(0)
        emb_out_avg = emb_out_avg / self.repeats

        self.edge_index = data.edge_index.cpu()
        self._att_coefs = (weights_first[0], att_weights_all)
        self._node_embs = emb_out_avg

        # Store text embeddings and projections for analysis
        if text_embeddings is not None:
            self._text_embeddings = text_embeddings.cpu().detach().numpy()

    def __enhanced_train(self, data, dgi_model, optimizer, encoder,
                         contrastive_proj_gnn=None, contrastive_proj_text=None,
                         text_embeddings=None):
        """Enhanced training step with optional contrastive learning"""
        dgi_model.train()
        if contrastive_proj_gnn is not None:
            contrastive_proj_gnn.train()
            contrastive_proj_text.train()

        optimizer.zero_grad()

        # DGI loss (original)
        pos_z, neg_z, summary = dgi_model(data)
        dgi_loss = dgi_model.loss(pos_z, neg_z, summary)

        total_loss = dgi_loss
        contrastive_loss = 0.0

        # Contrastive loss (NEW)
        if (self.loss_text_align > 0 and contrastive_proj_gnn is not None and
                text_embeddings is not None):
            # Get node embeddings from encoder
            node_embeddings = encoder.x_embs  # [N, hidden_dim]

            # Project both graph and text embeddings
            proj_graph = F.normalize(contrastive_proj_gnn(node_embeddings), p=2, dim=1)
            proj_text = F.normalize(contrastive_proj_text(text_embeddings.detach()), p=2, dim=1)

            # InfoNCE loss
            contrastive_loss = self.__info_nce_loss(proj_graph, proj_text, self.contrastive_temperature)

            total_loss = total_loss + self.loss_text_align * contrastive_loss

        total_loss.backward()
        optimizer.step()

        return float(total_loss.item()), float(
            contrastive_loss.item() if isinstance(contrastive_loss, torch.Tensor) else contrastive_loss)

    def __info_nce_loss(self, z_graph, z_text, temperature):
        """Compute InfoNCE contrastive loss between graph and text representations"""
        batch_size = z_graph.size(0)

        # Compute similarity matrix
        sim_matrix = torch.mm(z_graph, z_text.t()) / temperature  # [N, N]

        # Labels are diagonal (each node should be similar to its own text embedding)
        labels = torch.arange(batch_size, device=z_graph.device)

        # Cross entropy loss (InfoNCE)
        loss = F.cross_entropy(sim_matrix, labels)

        return loss

    def get_network(self,
                    keep_self_loops: bool = True,
                    edge_threshold_avgDegree: Optional[int] = 10,
                    edge_threshold_zscore: Optional[float] = None,
                    output_file: Optional[str] = None) -> nx.DiGraph:
        """
        The function is used to generate a predicted gene regulatory network (GRN) from the learned attention coefficients.

        Parameters:
            keep_self_loops (bool): A flag indicating whether to keep self-loops in the predicted network. Default is True.
            edge_threshold_avgDegree (Optional[int]): The average degree threshold for selecting edges.
                Only the top (N_nodes*edge_threshold_avgDegree) edges are kept. Default is 10.
            edge_threshold_zscore (Optional[float]): The z-score threshold for selecting edges.
                Only the edges with a z-score greater than edge_threshold_zscore are kept. Default is None.
                Ignored if 'edge_threshold_avgDegree' is not None.
            output_file (Optional[str]): The file path to save the predicted network. If None, the network is not saved. Default is None.

        Returns:
            (nx.DiGraph): The predicted gene regulatory network.
        """

        edge_index_ori = self.edge_index
        edge_index_with_selfloop, att_coefs_with_selfloop = self._att_coefs[0], self._att_coefs[1]

        ori_att_coefs_all = pd.DataFrame(
            {'from': edge_index_with_selfloop[0].numpy().astype(int),
             'to': edge_index_with_selfloop[1].numpy().astype(int),
             'att_coef_in': att_coefs_with_selfloop.mean(0, keepdim=False)[:, 0].numpy(),
             'att_coef_out': att_coefs_with_selfloop.mean(0, keepdim=False)[:, 1].numpy()}
        )
        ori_att_coefs_all['edge_idx_tmp'] = ori_att_coefs_all['from'].astype(str) + "|" \
                                            + ori_att_coefs_all['to'].astype(str)

        ## Scaled the attention coefficients for global ranking
        scaled_att_coefs = []
        g = nx.from_edgelist(edge_index_with_selfloop.numpy().T, create_using=nx.DiGraph)
        for i in range(2):
            # i==0: in-coming; i==1: out-going
            att_coef_i = att_coefs_with_selfloop[:, :, i]  # shape: [num_repeats, num_edges]

            # Edges are selected based on a global weight threshold.
            # Attention coefficients are scaled, which are seemed as edge weights,
            # by multiplying by the degree of central node.
            if i == 0:
                d = pd.DataFrame(g.in_degree(), columns=['index', 'degree'])
            else:  # i == 1
                d = pd.DataFrame(g.out_degree(), columns=['index', 'degree'])
            d.index = d['index']
            # att_coef_i * degree_i
            att_coef_i = att_coef_i * np.array(d.loc[edge_index_with_selfloop[1 - i, :].numpy(), 'degree'])
            att_coef_i = att_coef_i.t()  # shape: [num_edges, num_repeats]

            if not keep_self_loops:
                # remove all the self-loops
                edge_index, att_coef_i = remove_self_loops(edge_index_with_selfloop, att_coef_i)
            else:
                # only keep the self-loops from the prior network
                prior_selfloop_nodes = edge_index_ori[0, edge_index_ori[0] == edge_index_ori[1]]
                selfloop_not_in_prior = (edge_index_with_selfloop[0] == edge_index_with_selfloop[1]) & \
                                        ~(edge_index_with_selfloop[0][..., None] == prior_selfloop_nodes).any(-1)
                edge_index, att_coef_i = (edge_index_with_selfloop[:, ~selfloop_not_in_prior],
                                          att_coef_i[~selfloop_not_in_prior])

            # All edge weights without filtering
            # Shape: [2, num_edge_index, num_repeats]
            scaled_att_coefs = scaled_att_coefs + [att_coef_i.clone()]

        ## Combine the scaled attention coefficients of in-coming and out-going networks. 0: incoming; 1: outgoing
        scaled_att_coefs_combined = (scaled_att_coefs[0] * 0.5) + (scaled_att_coefs[1] * 0.5)
        scaled_att_coefs_all = pd.DataFrame(
            {'from': edge_index[0].numpy().astype(int),
             'to': edge_index[1].numpy().astype(int),
             'weights_in': scaled_att_coefs[0].mean(1, keepdim=False).numpy(),
             'weights_out': scaled_att_coefs[1].mean(1, keepdim=False).numpy(),
             'weights_combined': scaled_att_coefs_combined.mean(1, keepdim=False).numpy(),
             'weights_std': scaled_att_coefs_combined.std(1, keepdim=False).numpy()}
        )

        # used for debugging
        # ori_att_coefs_all = ori_att_coefs_all.loc[
        #     ori_att_coefs_all['edge_idx_tmp'].isin(
        #         scaled_att_coefs_all['from'].astype(str) + "|" + scaled_att_coefs_all['to'].astype(str)),
        # ].copy()
        # self.att_coef_all = pd.merge(ori_att_coef_all[['from', 'to', 'att_coef_in', 'att_coef_out']],
        #                              scaled_att_coef_all, on=['from', 'to'], how='inner')

        # Remove the noise edges. Make sure the coefficient of variation (CV) < 0.2.
        # Only use if the number of repeats no less than 10.
        if scaled_att_coefs[0].shape[1] >= 10:
            CV = scaled_att_coefs_all['weights_std'] / (scaled_att_coefs_all['weights_combined'] + 1e-9)
            cv_filter = (CV < 0.2)
        else:
            cv_filter = np.ones(scaled_att_coefs_combined.shape[0], dtype=bool)

        ## Select edges with a cutoff threshold
        att_weights_combined = scaled_att_coefs_all['weights_combined']
        if edge_threshold_avgDegree is not None:
            # Select top (N_nodes*edge_threshold_avgDegree) edges
            filtered_edge_idx = np.argsort(-att_weights_combined)[0:g.number_of_nodes() * edge_threshold_avgDegree]
            filtered_edge_idx = np.intersect1d(filtered_edge_idx, np.where(cv_filter)[0])
        else:
            if edge_threshold_zscore is not None:
                # Select top edges with z-score>edge_threshold_zscore
                m, s = att_weights_combined.mean(), att_weights_combined.std()
                edge_threshold_weight = m + (edge_threshold_zscore * s)
                filtered_edge_idx = np.where((att_weights_combined > edge_threshold_weight) & cv_filter)[0]
            else:  # All edges
                filtered_edge_idx = list(range(len(att_weights_combined)))
        scaled_att_coefs_filtered = scaled_att_coefs_all.iloc[filtered_edge_idx, :].copy()

        ## Output the scaled attention coefficient of the predicted network
        ori_att_coefs_filtered = ori_att_coefs_all.loc[
            ori_att_coefs_all['edge_idx_tmp'].isin(
                scaled_att_coefs_filtered['from'].astype(str) + "|" + scaled_att_coefs_filtered['to'].astype(str)),
            ['from', 'to', 'att_coef_in', 'att_coef_out']
        ].copy()
        net_filtered_df = pd.merge(scaled_att_coefs_filtered, ori_att_coefs_filtered, on=['from', 'to'], how='inner')

        ## To networkx
        G_nx = nx.from_pandas_edgelist(net_filtered_df, source='from', target='to', edge_attr=True,
                                       create_using=nx.DiGraph)
        num_nodes = len(set(edge_index_with_selfloop[0].numpy()).union(set(edge_index_with_selfloop[1].numpy())))
        largest_components = max(nx.weakly_connected_components(G_nx), key=len)
        if (len(largest_components) / num_nodes) < 0.5:
            print('  Warning: the size of maximal connected subgraph is less than half of the input whole graph!')
        G_nx = G_nx.subgraph(largest_components)

        ## Use gene name as index
        mappings = self._idx_GeneName_map.loc[self._idx_GeneName_map['idx'].isin(G_nx.nodes()), ['idx', 'geneName']]
        mappings = {idx: geneName for (idx, geneName) in np.array(mappings)}
        G_nx = nx.relabel_nodes(G_nx, mappings)
        self.GRN_predicted = G_nx

        ## Save the predicted network to file
        if isinstance(output_file, str):
            nx.write_edgelist(G_nx, output_file, delimiter=',', data=['weights_combined'])

        return G_nx

    def get_gene_embedding(self, output_file: Optional[str] = None) -> pd.DataFrame:
        """
        This function retrieves the gene embeddings learned by the network model.

        Parameters:
            output_file (Optional[str]): The file path to save the gene embeddings.
                If None, the gene embeddings are not saved. Default is None.

        Returns:
            (pd.DataFrame): A DataFrame containing the gene embeddings. The index of the DataFrame is the gene name.
        """

        if self.GRN_predicted is None:
            raise ValueError(
                f'Did not find the predicted network. Run `NetModel.get_network` first.'
            )
        emb = pd.DataFrame(self._node_embs, index=self._idx_GeneName_map['geneName'])
        emb = emb.loc[emb.index.isin(self.GRN_predicted.nodes), :]
        emb.index = emb.index.astype(str)

        if isinstance(output_file, str):
            emb.to_csv(output_file, index_label='geneName')

        return emb

    def get_TEDC2L_results(self,
                           keep_self_loops: bool = True,
                           edge_threshold_avgDegree: Optional[int] = 10,
                           edge_threshold_zscore: Optional[float] = None) -> TEDC2LResults:
        """
        This function retrieves the TEDC2L results, which include the predicted gene regulatory network (GRN)
        and the gene embeddings learned by the network model.

        Parameters:
            keep_self_loops (bool): A flag indicating whether to keep self-loops in the predicted network. Default is True.
            edge_threshold_avgDegree (int): The average degree threshold for selecting edges.
                Only the top (N_nodes*edge_threshold_avgDegree) edges are kept. Default is 10.
            edge_threshold_zscore (Optional[float]): The z-score threshold for selecting edges.
                Only the edges with a z-score greater than edge_threshold_zscore are kept. Default is None.
                Ignored if 'edge_threshold_avgDegree' is not None.

        Returns:
            (TEDC2LResults): An object containing the input data, predicted gene regulatory network (GRN) and the gene embeddings.
        """

        network = self.get_network(keep_self_loops, edge_threshold_avgDegree, edge_threshold_zscore)
        gene_embedding = self.get_gene_embedding()

        results = TEDC2LResults(adata=self._adata,
                                network=network,
                                gene_embedding=gene_embedding)

        return results
