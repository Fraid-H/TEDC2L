import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse, stats
import networkx as nx
import scanpy as sc
from typing import Optional, Union
from pathlib import Path
import re
import warnings

import matplotlib.pyplot as plt
from matplotlib.pyplot import gcf
import matplotlib.ticker as ticker
import seaborn as sns

from resources.tfs_data import TFs_human, TFs_mouse


def data_preparation(input_expData: Union[str, sc.AnnData, pd.DataFrame],
                     input_priorNet: Union[str, pd.DataFrame],
                     genes_DE: Optional[Union[str, pd.DataFrame]] = None,
                     additional_edges_pct: float = 0.01,
                     # New text similarity parameters with safeguards
                     text_knn: int = 0,
                     text_sim_tau: float = 0.0,
                     text_mix_alpha: float = -1.0,
                     text_max_edges: int = 0) -> dict[str: AnnData]:
    """
    Enhanced data preparation with text similarity-based edge addition and comprehensive safeguards.
    """

    print('[0] - Data loading and preprocessing...')

    ## [1] Single-cell RNA-seq data
    lineages = None
    if isinstance(input_expData, str):
        p = Path(input_expData)
        if p.suffix == '.csv':
            adata = sc.read_csv(input_expData, first_column_names=True)
            adata = adata.transpose()
        else:  # h5ad
            adata = sc.read_h5ad(input_expData)
    elif isinstance(input_expData, sc.AnnData):
        adata = input_expData
        lineages = adata.uns.get('lineages')
    elif isinstance(input_expData, pd.DataFrame):
        adata = sc.AnnData(X=input_expData)
    else:
        raise Exception("Invalid input! The input must be '.csv' format file or '.h5ad' "
                        "format file, or an 'AnnData' object!", input_expData)

    possible_species = 'mouse' if bool(re.search('[a-z]', adata.var_names[0])) else 'human'

    # Gene symbols are uniformly handled in uppercase
    adata.var_names = adata.var_names.str.upper()

    ## [2] Prior gene interaction network
    if isinstance(input_priorNet, str):
        netData = pd.read_csv(input_priorNet, index_col=None, header=0)
    elif isinstance(input_priorNet, pd.DataFrame):
        netData = input_priorNet.copy()
    else:
        raise Exception("Invalid input!", input_priorNet)

    # Make sure the input scRNA-seq data contains genes from the prior network
    netData['from'] = netData['from'].str.upper()
    netData['to'] = netData['to'].str.upper()
    netData = netData.loc[netData['from'].isin(adata.var_names.values)
                          & netData['to'].isin(adata.var_names.values), :]
    netData = netData.drop_duplicates(subset=['from', 'to'], keep='first', inplace=False)

    # Transfer into networkx object
    priori_network = nx.from_pandas_edgelist(netData, source='from', target='to', create_using=nx.DiGraph)
    priori_network_nodes = np.array(priori_network.nodes())

    # in_degree, out_degree (centrality)
    in_degree = pd.DataFrame.from_dict(nx.in_degree_centrality(priori_network),
                                       orient='index', columns=['in_degree'])
    out_degree = pd.DataFrame.from_dict(nx.out_degree_centrality(priori_network),
                                        orient='index', columns=['out_degree'])
    centrality = pd.concat([in_degree, out_degree], axis=1)
    centrality = centrality.loc[priori_network_nodes, :]

    ## [3] Create a mapper for node indices and gene names
    idx_GeneName_map = pd.DataFrame({'idx': range(len(priori_network_nodes)),
                                     'geneName': priori_network_nodes},
                                    index=priori_network_nodes)

    edgelist = pd.DataFrame({'from': idx_GeneName_map.loc[netData['from'].tolist(), 'idx'].tolist(),
                             'to': idx_GeneName_map.loc[netData['to'].tolist(), 'idx'].tolist()})

    ## [4] Add TF information
    is_TF = np.ones(len(priori_network_nodes), dtype=int)
    if possible_species == 'human':
        TFs_df = TFs_human
    else:
        TFs_df = TFs_mouse
    TF_list = TFs_df.iloc[:, 0].str.upper()
    is_TF[~np.isin(priori_network_nodes, TF_list)] = 0

    # Only keep the genes that exist in both single cell data and the prior gene interaction network
    adata = adata[:, priori_network_nodes]
    if lineages is None:
        cells_in_lineage_dict = {'all': adata.obs_names}
    else:
        cells_in_lineage_dict = {l: adata.obs_names[adata.obs[l].notna()] for l in lineages}

    print(f"Consider the input data with {len(cells_in_lineage_dict)} lineages:")
    if isinstance(genes_DE, str):
        genes_DE = pd.read_csv(genes_DE, index_col=0, header=0)

    if isinstance(genes_DE, pd.DataFrame) and (genes_DE.shape[1] != len(cells_in_lineage_dict)):
        raise Exception("The number of columns in 'genes_DE' does not match the number of lineages!", genes_DE)

    # Apply safeguards to text parameters
    N = len(priori_network_nodes)
    safe_text_knn = _safe_text_knn(N, text_knn)
    safe_text_sim_tau = _validate_similarity_threshold(text_sim_tau)
    safe_text_max_edges = max(0, text_max_edges)

    adata_lineages = dict()
    for li, (l, c) in enumerate(cells_in_lineage_dict.items()):
        print(f"  Lineage - {l}:")

        adata_l = sc.AnnData(X=adata[c, :].to_df())
        adata_l.var['is_TF'] = is_TF
        adata_l.varm['centrality_prior_net'] = centrality
        adata_l.varm['idx_GeneName_map'] = idx_GeneName_map
        adata_l.uns['name'] = l

        ## [5] Enhanced edge addition with comprehensive safeguards
        ori_edgeNum = len(edgelist)

        # Initialize dataframes for different edge types
        df_scc = pd.DataFrame(columns=['from', 'to', 'w_scc'])
        df_text = pd.DataFrame(columns=['from', 'to', 'w_text'])
        df_extra = pd.DataFrame(columns=['from', 'to'])

        # Spearman correlation edges (original logic)
        if additional_edges_pct > 0:
            try:
                if isinstance(adata_l.X, sparse.csr_matrix):
                    gene_exp = pd.DataFrame(adata_l.X.A.T, index=priori_network_nodes)
                else:
                    gene_exp = pd.DataFrame(adata_l.X.T, index=priori_network_nodes)

                SCC, _ = stats.spearmanr(gene_exp, axis=1)
                edges_corr = np.absolute(SCC)
                np.fill_diagonal(edges_corr, 0.0)
                x, y = np.where(edges_corr > 0.6)
                df_scc = pd.DataFrame({'from': x, 'to': y, 'w_scc': edges_corr[x, y]})

                # Calculate target number of additional edges
                total_possible_edges = gene_exp.shape[0] * (gene_exp.shape[0] - 1)
                addi_top_k = int(total_possible_edges * additional_edges_pct)

            except Exception as e:
                print(f"    Warning: Spearman correlation calculation failed: {e}")
                df_scc = pd.DataFrame(columns=['from', 'to', 'w_scc'])
                addi_top_k = 0
        else:
            addi_top_k = 0

        # Text similarity edges (NEW with safeguards)
        text_edges_added = 0
        if ('gene_text_emb' in adata_l.varm and
                (safe_text_knn > 0 or safe_text_sim_tau > 0)):

            print(f"    Adding text similarity edges...")
            try:
                gene_text_emb = adata_l.varm['gene_text_emb']

                # Validate embeddings
                is_valid, warning_msg = _validate_text_embeddings(gene_text_emb, priori_network_nodes)
                if warning_msg != "Text embeddings appear valid":
                    print(f"    Warning: {warning_msg}")

                if is_valid or np.any(gene_text_emb != 0):  # Proceed if valid or has non-zero values
                    # Generate text similarity edges using safer helper function
                    df_text = _compute_text_similarity_edges_safe(
                        gene_text_emb=gene_text_emb,
                        text_knn=safe_text_knn,
                        text_sim_tau=safe_text_sim_tau,
                        text_max_edges=safe_text_max_edges
                    )
                    text_edges_added = len(df_text)
                else:
                    print(f"    Skipping text similarity due to invalid embeddings")

            except Exception as e:
                print(f"    Warning: Text similarity calculation failed: {e}")
                df_text = pd.DataFrame(columns=['from', 'to', 'w_text'])
                text_edges_added = 0

        # Combine edges using specified strategy with error handling
        if additional_edges_pct > 0 or text_edges_added > 0:
            try:
                if (text_mix_alpha >= 0 and len(df_scc) > 0 and len(df_text) > 0):
                    # Mixed ranking strategy
                    print(f"    Using mixed ranking strategy (alpha={text_mix_alpha})...")
                    df_extra = _combine_edges_mixed_ranking(df_scc, df_text, text_mix_alpha,
                                                            addi_top_k if additional_edges_pct > 0 else None)
                else:
                    # Union strategy (default)
                    print(f"    Using union strategy...")
                    df_extra = _combine_edges_union(df_scc, df_text, addi_top_k if additional_edges_pct > 0 else None)

            except Exception as e:
                print(f"    Warning: Edge combination failed: {e}, using union fallback")
                df_extra = _combine_edges_union(df_scc, df_text, addi_top_k if additional_edges_pct > 0 else None)

        # Combine with prior edges
        if len(df_extra) > 0:
            edgelist_new = pd.concat([edgelist, df_extra], ignore_index=True)
            edgelist_new = edgelist_new.drop_duplicates(subset=['from', 'to'], keep='first')
        else:
            edgelist_new = edgelist.copy()

        # Validate edge counts
        final_edgelist = edgelist_new
        adata_l.uns['edgelist'] = final_edgelist

        # Validate and log edge addition
        edge_validation = _validate_edge_counts(
            ori_edgeNum, len(df_scc), text_edges_added,
            len(final_edgelist), additional_edges_pct
        )

        if edge_validation['warnings']:
            for warning in edge_validation['warnings']:
                print(f"    Warning: {warning}")

        # Log edge addition statistics
        total_new_edges = len(final_edgelist) - ori_edgeNum
        print(f'    Spearman correlation edges: {len(df_scc)}')
        print(f'    Text similarity edges: {text_edges_added}')
        print(f'    Total new edges added: {total_new_edges}')
        print(f'    Final edge count: {len(final_edgelist)}')

        ## [6] Add differential expression scores to the objects
        logFC = adata.var.get(l + '_logFC')
        if (genes_DE is None) and (logFC is None):
            pass
        else:
            if logFC is not None:
                genes_DE_lineage = logFC
            else:
                genes_DE_lineage = genes_DE.iloc[:, li]

            genes_DE_lineage = pd.DataFrame(genes_DE_lineage).iloc[:, 0]
            genes_DE_lineage.index = genes_DE_lineage.index.str.upper()
            genes_DE_lineage = genes_DE_lineage[genes_DE_lineage.index.isin(priori_network_nodes)].abs().dropna()
            node_score_auxiliary = pd.Series(np.zeros(len(priori_network_nodes)), index=priori_network_nodes)
            node_score_auxiliary[genes_DE_lineage.index] = genes_DE_lineage.values
            node_score_auxiliary = np.array(node_score_auxiliary)
            adata_l.var['node_score_auxiliary'] = node_score_auxiliary

        adata_lineages[l] = adata_l
        print(f"    n_genes × n_cells = {adata_l.n_vars} × {adata_l.n_obs}")

    return adata_lineages


# Helper functions with comprehensive safeguards
def _safe_text_knn(N: int, text_knn: int) -> int:
    """Apply safety limits to text_knn parameter."""
    if text_knn <= 0:
        return 0

    max_k_absolute = 100
    max_k_adaptive = max(10, min(int(0.01 * N), 50))
    max_k_available = N - 1

    safe_k = min(text_knn, max_k_absolute, max_k_adaptive, max_k_available)

    if safe_k != text_knn:
        warnings.warn(f"Adjusted text_knn from {text_knn} to {safe_k} for stability")

    return safe_k


def _validate_similarity_threshold(tau: float) -> float:
    """Validate and adjust similarity threshold."""
    if tau < 0:
        warnings.warn(f"Negative similarity threshold {tau} set to 0")
        return 0.0
    if tau > 1.0:
        warnings.warn(f"Similarity threshold {tau} > 1.0, set to 1.0")
        return 1.0
    return tau


def _validate_text_embeddings(embeddings: np.ndarray, gene_names: list) -> tuple:
    """Validate text embeddings for potential issues."""
    issues = []

    if np.any(np.isnan(embeddings)):
        issues.append("Contains NaN values")
    if np.any(np.isinf(embeddings)):
        issues.append("Contains infinite values")

    zero_genes = np.all(embeddings == 0, axis=1).sum()
    if zero_genes > len(gene_names) * 0.5:
        issues.append(f"{zero_genes}/{len(gene_names)} genes have zero embeddings")

    warning_msg = "; ".join(issues) if issues else "Text embeddings appear valid"
    return len(issues) == 0, warning_msg


def _compute_text_similarity_edges_safe(gene_text_emb: np.ndarray,
                                        text_knn: int = 0,
                                        text_sim_tau: float = 0.0,
                                        text_max_edges: int = 0,
                                        chunk_size: int = 1000) -> pd.DataFrame:
    """Memory-safe computation of text similarity edges."""
    N = gene_text_emb.shape[0]

    if N == 0 or np.all(gene_text_emb == 0):
        return pd.DataFrame(columns=['from', 'to', 'w_text'])

    # Determine safe chunk size
    memory_per_chunk_mb = (chunk_size * N * 4) / (1024 * 1024)
    if memory_per_chunk_mb > 500:  # Keep under 500MB per chunk
        safe_chunk_size = max(1, int((500 * 1024 * 1024) / (N * 4)))
        print(f"      Adjusted chunk_size from {chunk_size} to {safe_chunk_size} for memory efficiency")
        chunk_size = safe_chunk_size

    # L2 normalize embeddings
    norms = np.linalg.norm(gene_text_emb, axis=1, keepdims=True) + 1e-9
    embeddings_norm = gene_text_emb / norms

    text_edges = []

    try:
        if text_knn > 0:
            # Top-k mode with chunked processing
            print(f"      Computing text similarity top-{text_knn} edges...")
            for i in range(0, N, chunk_size):
                end_i = min(i + chunk_size, N)
                chunk_emb = embeddings_norm[i:end_i]

                sim_matrix = chunk_emb @ embeddings_norm.T

                # Exclude self-loops
                for j, global_idx in enumerate(range(i, end_i)):
                    sim_matrix[j, global_idx] = -1.0

                # Find top-k
                k_actual = min(text_knn, N - 1)
                if k_actual > 0:
                    top_k_indices = np.argpartition(-sim_matrix, kth=k_actual - 1, axis=1)[:, :k_actual]

                    for j, global_idx in enumerate(range(i, end_i)):
                        neighbors = top_k_indices[j]
                        similarities = sim_matrix[j, neighbors]

                        for neighbor_idx, sim_score in zip(neighbors, similarities):
                            if sim_score > 0:
                                text_edges.append([global_idx, neighbor_idx, sim_score])

        elif text_sim_tau > 0:
            # Threshold mode with chunked processing
            print(f"      Computing text similarity threshold edges (tau={text_sim_tau})...")
            for i in range(0, N, chunk_size):
                end_i = min(i + chunk_size, N)
                chunk_emb = embeddings_norm[i:end_i]

                sim_matrix = chunk_emb @ embeddings_norm.T

                # Exclude self-loops
                for j, global_idx in enumerate(range(i, end_i)):
                    sim_matrix[j, global_idx] = 0.0

                # Find edges above threshold
                rows, cols = np.where(sim_matrix >= text_sim_tau)
                for r, c in zip(rows, cols):
                    global_r = i + r
                    text_edges.append([global_r, c, sim_matrix[r, c]])

    except MemoryError as e:
        print(f"      Memory error in text similarity computation: {e}")
        print(f"      Falling back to smaller chunk size")
        # Retry with smaller chunk size
        return _compute_text_similarity_edges_safe(gene_text_emb, text_knn, text_sim_tau,
                                                   text_max_edges, chunk_size // 2)

    if not text_edges:
        return pd.DataFrame(columns=['from', 'to', 'w_text'])

    # Convert to DataFrame and apply limits
    df_text = pd.DataFrame(text_edges, columns=['from', 'to', 'w_text'])
    df_text = df_text.drop_duplicates(subset=['from', 'to'])

    # Apply max edges limit if specified
    if text_max_edges > 0 and len(df_text) > text_max_edges:
        df_text = df_text.nlargest(text_max_edges, 'w_text')
        print(f"      Limited to top {text_max_edges} text similarity edges")

    print(f"      Generated {len(df_text)} text similarity edges")
    return df_text


def _combine_edges_mixed_ranking(df_scc: pd.DataFrame, df_text: pd.DataFrame,
                                 text_mix_alpha: float, addi_top_k: Optional[int]) -> pd.DataFrame:
    """Combine edges using mixed ranking strategy."""
    df_cand = pd.concat([df_scc[['from', 'to', 'w_scc']],
                         df_text[['from', 'to', 'w_text']]], ignore_index=True)

    df_cand = df_cand.groupby(['from', 'to'], as_index=False).agg({
        'w_scc': 'max', 'w_text': 'max'
    })

    df_cand['w_scc'] = df_cand['w_scc'].fillna(0.0)
    df_cand['w_text'] = df_cand['w_text'].fillna(0.0)

    # Normalize scores to [0,1]
    for col in ['w_scc', 'w_text']:
        if df_cand[col].max() > 0:
            vmin, vmax = df_cand[col].min(), df_cand[col].max()
            df_cand[col] = (df_cand[col] - vmin) / (vmax - vmin + 1e-9)

    df_cand['w_mix'] = text_mix_alpha * df_cand['w_scc'] + (1.0 - text_mix_alpha) * df_cand['w_text']

    df_cand = df_cand.sort_values('w_mix', ascending=False)
    if addi_top_k is not None and len(df_cand) > addi_top_k:
        df_cand = df_cand.head(addi_top_k)

    return df_cand[['from', 'to']].copy()


def _combine_edges_union(df_scc: pd.DataFrame, df_text: pd.DataFrame,
                         addi_top_k: Optional[int]) -> pd.DataFrame:
    """Combine edges using union strategy."""
    candidates = []

    # Add Spearman correlation edges
    if len(df_scc) > 0:
        df_scc_sorted = df_scc.sort_values('w_scc', ascending=False)
        if addi_top_k is not None and len(df_scc_sorted) > addi_top_k:
            df_scc_sorted = df_scc_sorted.head(addi_top_k)
        candidates.append(df_scc_sorted[['from', 'to']])

    # Add text similarity edges
    if len(df_text) > 0:
        candidates.append(df_text[['from', 'to']])

    # Combine and deduplicate
    if candidates:
        df_extra = pd.concat(candidates, ignore_index=True)
        df_extra = df_extra.drop_duplicates(subset=['from', 'to'])
        return df_extra
    else:
        return pd.DataFrame(columns=['from', 'to'])


def _validate_edge_counts(original_edges: int, spearman_edges: int,
                          text_edges: int, final_edges: int,
                          expected_additional_pct: float) -> dict:
    """Validate edge addition patterns."""
    results = {
        'original_edges': original_edges,
        'spearman_edges': spearman_edges,
        'text_edges': text_edges,
        'final_edges': final_edges,
        'net_added': final_edges - original_edges,
        'warnings': []
    }

    if final_edges < original_edges:
        results['warnings'].append("Final edge count is less than original")

    max_reasonable = original_edges * (1 + expected_additional_pct * 10)
    if final_edges > max_reasonable:
        results['warnings'].append(f"Excessive edge addition: {final_edges} vs max {max_reasonable}")

    return results


# Rest of the utility functions remain unchanged...
# (cluster_cell_by_RGM, network_topological_properties, prepare_data_for_R,
#  process_Slingshot_MAST_R, process_MAST_R, plot_controllability_metrics)


def _compute_text_similarity_edges(gene_text_emb: np.ndarray,
                                   text_knn: int = 0,
                                   text_sim_tau: float = 0.0,
                                   text_max_edges: int = 0,
                                   chunk_size: int = 1000) -> pd.DataFrame:
    """
    Compute text similarity-based edges using cosine similarity.

    Parameters:
        gene_text_emb: Gene text embeddings [N, d]
        text_knn: Number of top-k neighbors (0 = disabled)
        text_sim_tau: Similarity threshold (0 = disabled, only used when text_knn=0)
        text_max_edges: Maximum number of edges to return (0 = unlimited)
        chunk_size: Chunk size for memory-efficient computation

    Returns:
        DataFrame with columns ['from', 'to', 'w_text']
    """
    N = gene_text_emb.shape[0]

    # Check for valid embeddings
    if N == 0 or np.all(gene_text_emb == 0):
        return pd.DataFrame(columns=['from', 'to', 'w_text'])

    # L2 normalize embeddings
    norms = np.linalg.norm(gene_text_emb, axis=1, keepdims=True) + 1e-9
    embeddings_norm = gene_text_emb / norms

    text_edges = []

    if text_knn > 0:
        # Top-k mode: for each gene, find k most similar genes
        print(f"      Computing text similarity top-{text_knn} edges...")

        # Limit k to reasonable value
        k_actual = min(text_knn, max(10, int(0.01 * N)), N - 1)
        if k_actual != text_knn:
            print(f"      Adjusted text_knn from {text_knn} to {k_actual} for stability")

        # Process in chunks to avoid memory issues
        for i in range(0, N, chunk_size):
            end_i = min(i + chunk_size, N)
            chunk_emb = embeddings_norm[i:end_i]  # [chunk_size, d]

            # Compute similarity with all genes
            sim_matrix = chunk_emb @ embeddings_norm.T  # [chunk_size, N]

            # Set self-similarity to -1 to exclude self-loops in top-k
            for j, global_idx in enumerate(range(i, end_i)):
                sim_matrix[j, global_idx] = -1.0

            # Find top-k for each gene in chunk
            top_k_indices = np.argpartition(-sim_matrix, kth=k_actual - 1, axis=1)[:, :k_actual]

            # Create edges for this chunk
            for j, global_idx in enumerate(range(i, end_i)):
                neighbors = top_k_indices[j]
                similarities = sim_matrix[j, neighbors]

                for neighbor_idx, sim_score in zip(neighbors, similarities):
                    if sim_score > 0:  # Only positive similarities
                        text_edges.append([global_idx, neighbor_idx, sim_score])

    elif text_sim_tau > 0:
        # Threshold mode: keep all edges above threshold
        print(f"      Computing text similarity threshold edges (tau={text_sim_tau})...")

        # Process in chunks to avoid memory issues
        for i in range(0, N, chunk_size):
            end_i = min(i + chunk_size, N)
            chunk_emb = embeddings_norm[i:end_i]

            # Compute similarity with all genes
            sim_matrix = chunk_emb @ embeddings_norm.T  # [chunk_size, N]

            # Set self-similarity to 0 to exclude self-loops
            for j, global_idx in enumerate(range(i, end_i)):
                sim_matrix[j, global_idx] = 0.0

            # Find edges above threshold
            rows, cols = np.where(sim_matrix >= text_sim_tau)
            for r, c in zip(rows, cols):
                global_r = i + r
                text_edges.append([global_r, c, sim_matrix[r, c]])

    if not text_edges:
        return pd.DataFrame(columns=['from', 'to', 'w_text'])

    # Convert to DataFrame
    df_text = pd.DataFrame(text_edges, columns=['from', 'to', 'w_text'])
    df_text = df_text.drop_duplicates(subset=['from', 'to'])  # Remove any duplicates

    # Apply max edges limit if specified
    if text_max_edges > 0 and len(df_text) > text_max_edges:
        df_text = df_text.nlargest(text_max_edges, 'w_text')
        print(f"      Limited to top {text_max_edges} text similarity edges")

    print(f"      Generated {len(df_text)} text similarity edges")
    return df_text


def cluster_cell_by_RGM(auc_mtx, true_cell_label, method='ward', k=None):
    """
    Cluster cells based on RGM activity matrix by using hierarchical clustering.
    """
    assert method in {'ward', 'complete', 'average', 'single'}
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score, normalized_mutual_info_score, adjusted_rand_score

    auc_mtx_Z = pd.DataFrame(index=auc_mtx.index, columns=list(auc_mtx.columns))
    for row in list(auc_mtx.index):
        auc_mtx_Z.loc[row, :] = (auc_mtx.loc[row, :] - auc_mtx.loc[row, :].mean()) / auc_mtx.loc[row, :].std(ddof=0)

    if k is not None:
        ac = AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage=method).fit(auc_mtx_Z)
        predicted_cell_label = ac.labels_
        NMIs = normalized_mutual_info_score(true_cell_label, predicted_cell_label)
        ARIs = adjusted_rand_score(true_cell_label, predicted_cell_label)
        Silhouettes = silhouette_score(auc_mtx, predicted_cell_label, metric='euclidean')
        N_clus = k
    else:
        NMIs, ARIs, N_clus, Silhouettes = [], [], [], []
        max_cluster_num = len(set(true_cell_label)) * 2
        for i in range(2, max_cluster_num):
            ac = AgglomerativeClustering(n_clusters=i, affinity='euclidean', linkage=method).fit(auc_mtx_Z)
            out = ac.labels_
            NMIs.append(normalized_mutual_info_score(true_cell_label, out))
            ARIs.append(adjusted_rand_score(true_cell_label, out))
            N_clus.append(i)
            Silhouettes.append(silhouette_score(auc_mtx, out, metric='euclidean'))

    return {'NMIs': NMIs, 'ARIs': ARIs, 'Silhouettes': Silhouettes, 'num_clusters': N_clus}


def network_topological_properties(prior_network: pd.DataFrame):
    """
    Compute the basic topology metrics of the input network.
    """
    import networkx as nx
    from sklearn.linear_model import LinearRegression as lr

    prior_nx = nx.from_pandas_edgelist(prior_network, source='from', target='to', edge_attr=None,
                                       create_using=nx.DiGraph)

    # Number of genes
    N_genes = prior_nx.number_of_nodes()

    # Number of edges
    N_edges = len(prior_network[['from', 'to']].drop_duplicates())

    # Number of source genes
    N_source_genes = len(prior_network['from'].unique())

    # Number of target genes
    N_target_genes = len(prior_network['to'].unique())

    # Density
    Density = nx.density(prior_nx)

    # Average degree / in_degree / out_degree
    degree = sum(dict(prior_nx.degree()).values()) / N_genes

    # Clustering coefficient (time-consuming)
    clustering_coefficient = nx.average_clustering(prior_nx)

    # Slope of degree distribution
    degree_sequence = pd.DataFrame(np.array(prior_nx.degree))
    degree_sequence.columns = ["ind", "degree"]
    degree_sequence = degree_sequence.set_index("ind")
    dist = degree_sequence.degree.value_counts() / degree_sequence.degree.value_counts().sum()
    dist.index = dist.index.astype(int)

    x = np.log(dist.index.values).reshape([-1, 1])
    y = np.log(dist.values).reshape([-1, 1])

    model = lr()
    model.fit(x, y)

    slope = model.coef_[0][0]

    print(  # f"Dataset-species: {dataset}-{species}\n"
        f"Number of genes: {N_genes}\nNumber of edges: {N_edges}\n"
        f"Number of source genes: {N_source_genes}\nNumber of target genes: {N_target_genes}\n"
        f"Network density: {Density}\nAverage degree: {degree:.4f}\n"
        f"Average clustering coefficient: {clustering_coefficient:.4f}\n"
        f"Slope of degree distribution: {slope:.4f}")


def prepare_data_for_R(adata: sc.AnnData,
                       temp_R_dir: str,
                       reducedDim: Optional[str] = None,
                       cluster_label: Optional[str] = None):
    """
    Process the AnnData object and save the necessary data to files.
    These data files are prepared for running the `slingshot_MAST_script.R` or `MAST_script.R` scripts.
    """
    if 'log_transformed' not in adata.layers:
        raise ValueError(
            f'Did not find `log_transformed` in adata.layers.'
        )

    if isinstance(adata.layers['log_transformed'], sparse.csr_matrix):
        exp_normalized = adata.layers['log_transformed'].A
    else:
        exp_normalized = adata.layers['log_transformed']

    # The normalized and log transformed data is used for MAST
    normalized_counts = pd.DataFrame(exp_normalized,
                                     index=adata.obs_names,
                                     columns=adata.var_names)
    normalized_counts.to_csv(temp_R_dir + '/exp_normalized.csv', sep=',')

    # The reduced dimension data is used for Slingshot
    if reducedDim is not None:
        reducedDim_data = pd.DataFrame(adata.obsm[reducedDim], dtype='float32', index=None)
        reducedDim_data.to_csv(temp_R_dir + '/data_reducedDim.csv', index=None)
    else:
        if 'lineages' not in adata.uns:
            raise ValueError(
                f'Did not find `lineages` in adata.uns.'
            )
        else:
            pseudotime_all = pd.DataFrame(index=adata.obs_names)
            for li in adata.uns['lineages']:
                pseudotime_all[li] = adata.obs[li]
            pseudotime_all.to_csv(temp_R_dir + '/pseudotime_lineages.csv', index=True)

    # Cluster Labels (Leiden)
    if cluster_label is not None:
        cluster_labels = pd.DataFrame(adata.obs[cluster_label])
        cluster_labels.to_csv(temp_R_dir + '/clusters.csv')


def process_Slingshot_MAST_R(temp_R_dir: str,
                             split_num: int = 4,
                             start_cluster: int = 0,
                             end_cluster: Optional[list] = None):
    """
    Run the `slingshot_MAST_script.R` to get pseudotim and differential expression information for each lineage.
    """
    import subprocess
    import importlib.resources as res

    R_script_path = 'slingshot_MAST_script.R'
    with res.path('TEDC2L', R_script_path) as datafile:
        R_script_path = datafile

    path = Path(temp_R_dir)
    path.mkdir(exist_ok=path.exists(), parents=True)

    args = f'Rscript {R_script_path} {temp_R_dir} {split_num} {start_cluster}'
    if end_cluster is not None:
        args += f' {end_cluster}'
    print('Running Slingshot and MAST using: \'{}\'\n'.format(args))
    print('It will take a few minutes ...')
    with subprocess.Popen(args,
                          stdout=None, stderr=subprocess.PIPE,
                          shell=True) as p:
        out, err = p.communicate()
        if p.returncode == 0:
            print(f'Done. The results are saved in \'{temp_R_dir}\'.')
            print(f'      Trajectory (pseudotime) information: \'pseudotime_lineages.csv\'.')
            lineages = pd.read_csv(temp_R_dir + '/pseudotime_lineages.csv', index_col=0)
            lineages.columns
            print('      Differential expression information: ', end='')
            for l in lineages.columns:
                print(f'\'DEgenes_MAST_sp{split_num}_{l}.csv\' ', end='')
        else:
            print(f'Something error: returncode={p.returncode}.')


def process_MAST_R(temp_R_dir: str, split_num: int = 4):
    """
    Run the `MAST_script.R` to get differential expression information for each lineage.
    """
    import subprocess
    import importlib.resources as res

    R_script_path = 'MAST_script.R'
    with res.path('TEDC2L', R_script_path) as datafile:
        R_script_path = datafile

    path = Path(temp_R_dir)
    path.mkdir(exist_ok=path.exists(), parents=True)

    args = f'Rscript {R_script_path} {temp_R_dir} {split_num}'
    print('Running MAST using: \'{}\'\n'.format(args))
    print('It will take a few minutes ...')
    with subprocess.Popen(args,
                          stdout=None, stderr=subprocess.PIPE,
                          shell=True) as p:
        out, err = p.communicate()
        if p.returncode == 0:
            print(f'Done. The results are saved in \'{temp_R_dir}\'.')
            print(f'      Differential expression information: \'DEgenes_MAST_sp{split_num}_<x>.csv\'')
        else:
            print(f'Something error: returncode={p.returncode}.')


def plot_controllability_metrics(TEDC2L_results: Union[dict, list], return_value: bool = False):
    """
    Plot bar charts for controllability analyses.
    """
    # calculate metrics
    con_df = pd.DataFrame(columns=['MDS_controllability_score', 'MFVS_controllability_score',
                                   'Jaccard_index', 'Driver_regulators_coverage', 'Lineage'])
    for k in TEDC2L_results:
        if isinstance(k, str):
            result = TEDC2L_results[k]
        else:
            result = k
        drivers_df = result.driver_regulator

        MFVS_driver_set = set(drivers_df.loc[drivers_df['is_MFVS_driver']].index)
        MDS_driver_set = set(drivers_df.loc[drivers_df['is_MDS_driver']].index)
        driver_regulators = set(drivers_df.loc[drivers_df['is_driver_regulator']].index)
        top_ranked_genes = driver_regulators.union(
            (set(drivers_df.index) - MFVS_driver_set.union(MDS_driver_set)))
        N_genes = result.n_genes

        # MDS controllability score
        MDS_con = 1 - len(MDS_driver_set) / N_genes
        # MFVS controllability score
        MFVS_con = 1 - len(MFVS_driver_set) / N_genes
        # Jaccard index
        Jaccard_con = len(MDS_driver_set.intersection(MFVS_driver_set)) / len(MDS_driver_set.union(MFVS_driver_set))
        # driver regulators coverage
        Critical_con = len(driver_regulators) / len(MDS_driver_set.union(MFVS_driver_set))

        con_df.loc[len(con_df)] = {'MDS_controllability_score': MDS_con,
                                   'MFVS_controllability_score': MFVS_con,
                                   'Jaccard_index': Jaccard_con,
                                   'Driver_regulators_coverage': Critical_con,
                                   'Lineage': result.name}

    con_df = pd.melt(con_df, id_vars=['Lineage'])

    # plot
    fig = plt.figure(figsize=(9, 2))
    sns.set_theme(style="ticks", font_scale=1.0)
    surrent_palette = sns.color_palette("Set1")

    # Controallability score
    con_df1 = con_df.loc[con_df['variable'].isin(['MDS_controllability_score', 'MFVS_controllability_score'])]
    fig.add_subplot(1, 3, 1)
    ax1 = sns.barplot(x="variable", y="value", hue="Lineage", palette=surrent_palette,
                      data=con_df1)
    ax1.set_xlabel('')
    ax1.set_xticklabels(['MDS', 'MFVS'], rotation=45, ha="right")
    ax1.set_ylabel('Controllability Score')
    ax1.set_ylim(con_df1['value'].min().round(1)-0.1, 1.0)
    sns.despine()
    ax1.get_legend().remove()

    # Jaccard index
    con_df2 = con_df.loc[con_df['variable'].isin(['Jaccard_index'])]
    fig.add_subplot(1, 3, 2)
    ax2 = sns.barplot(x="variable", y="value", hue="Lineage", palette=surrent_palette,
                      data=con_df2)
    ax2.set_xlabel('')
    ax2.set_xticklabels('')
    ax2.set_ylabel(r'Jaccard Index\nbetween MFVS & MDS')
    sns.despine()
    ax2.get_legend().remove()

    # Driver regulators coverage
    con_df3 = con_df.loc[con_df['variable'].isin(['Driver_regulators_coverage'])]
    fig.add_subplot(1, 3, 3)
    ax3 = sns.barplot(x="variable", y="value", hue="Lineage", palette=surrent_palette,
                      data=con_df3)
    ax3.set_xlabel('')
    ax3.set_xticklabels('')
    ax3.set_ylabel('Driver Regulators Coverage')
    sns.despine()
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')

    plt.subplots_adjust(wspace=0.45)

    if return_value:
        return con_df
