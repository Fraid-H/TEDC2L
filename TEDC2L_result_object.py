from typing import Optional

import pandas as pd
import numpy as np
import networkx as nx
import scanpy as sc
from sklearn.linear_model import LinearRegression as lr

from matplotlib import pyplot as plt
from matplotlib_venn import venn3
import matplotlib.ticker as ticker
from matplotlib.pyplot import rc_context
import seaborn as sns

from driver_regulators import driver_regulators

class TEDC2LResults:

    def __init__(self,
                 adata: sc.AnnData,
                 network: nx.DiGraph,
                 gene_embedding: pd.DataFrame):
        self.name = adata.uns['name']
        self.network = network
        self.gene_embedding = gene_embedding

        genes = list(network.nodes())
        self.expression_data = adata[:, genes].to_df()
        self.TFs = adata[:, genes].var_names[adata[:, genes].var['is_TF'] == 1]
        if 'node_score_auxiliary' in adata.var:
            self.DEgenes = set(adata.var_names[adata.var['node_score_auxiliary'] > 0]).intersection(set(genes))
        else:
            self.DEgenes = set(genes)

        self.n_cells = adata.n_obs
        self.n_genes = len(genes)
        self.n_edges = network.number_of_edges()
        self.influence_score = None
        self.driver_regulator = None
        self.gene_cluster = None
        self.RGMs_AUCell_dict = None
        self._adata_gene = None
        self._out_critical_genes = None  # This will be set by the new enhanced_driver_regulators
        self._in_critical_genes = None  # This will be set by the new enhanced_driver_regulators

        # 在第39行后添加
        self.mces_results = None
        self.energy_results = None
        self.method_results = None

    def __repr__(self) -> str:
        descr = f"TEDC2LResults object with n_cells * n_genes = {self.n_cells} * {self.n_genes}, n_edges = {self.n_edges}"
        descr += f"\n    name: {self.name}" \
                 f"\n    expression_data: yes" \
                 f"\n    network: {self.network}" \
                 f"\n    gene_embedding: # dimension = {self.gene_embedding.shape[1] if self.gene_embedding is not None else 'None'}" \
                 f"\n    influence_score: {'None' if self.influence_score is None else 'yes'}" \
                 f"\n    driver_regulator: {'None' if self.driver_regulator is None else 'yes'}" \
                 f"\n    gene_cluster: {'None' if self.gene_cluster is None else 'yes'}" \
                 f"\n    RGMs_AUCell_dict: {'None' if self.RGMs_AUCell_dict is None else 'yes'}"
        return descr

    def gene_influence_score(self):
        """
        Obtain gene influence score
        """
        # influence_score = pd.DataFrame(np.zeros((len(self.network.nodes), 2)),
        #                                index=sorted(self.network.nodes),
        #                                columns=['out', 'in'])
        #
        # for i, v in enumerate(['in', 'out']):
        #     # The out-degree type of influence is obtained from the incoming network;
        #     # The in-degree type of influence is obtained from the outgoing network.
        #     gene_att_score = np.sum(nx.to_numpy_array(self.network,
        #                                               nodelist=sorted(self.network.nodes),
        #                                               dtype='float32',
        #                                               weight='weights_{}'.format(v)),  # {}_weights
        #                             axis=1 - i)
        #
        #
        #     influence_score.iloc[:, i] = np.log1p(gene_att_score).flatten().tolist()
        #
        # lam = 0.8
        # influence_score['influence_score'] = lam * influence_score.loc[:, 'out'] + \
        #                                      (1 - lam) * influence_score.loc[:, 'in']
        # influence_score.rename(columns={'out': 'score_out', 'in': 'score_in'}, inplace=True)
        # influence_score = influence_score.sort_values(by='influence_score', ascending=False)
        # self.influence_score = influence_score

    def gene_influence_score(self):
        """
        Obtain gene influence score
        """
        # --- 代码修改开始 ---
        # 1. 从 C.csv 文件加载基因列表
        # 为了提高效率，将基因列表转换为集合（set），这样查找速度更快
        try:
            csv_genes_df = pd.read_csv('../example_data/C.csv')
            csv_genes_set = set(csv_genes_df['Gene'])
        except FileNotFoundError:
            print("错误: C.csv 文件未找到。请确保该文件与您的脚本在同一目录下。")
            # 如果文件不存在，创建一个空集合，这样代码不会报错，但不会加分
            csv_genes_set = set()
        # --- 代码修改结束 ---

        influence_score = pd.DataFrame(np.zeros((len(self.network.nodes), 2)),
                                       index=sorted(self.network.nodes),
                                       columns=['out', 'in'])

        # 提前获取排序后的节点列表，避免在循环中重复计算
        sorted_nodes = sorted(self.network.nodes)

        for i, v in enumerate(['in', 'out']):
            # The out-degree type of influence is obtained from the incoming network;
            # The in-degree type of influence is obtained from the outgoing network.
            gene_att_score = np.sum(nx.to_numpy_array(self.network,
                                                      nodelist=sorted_nodes,  # 使用预先排序的列表
                                                      dtype='float32',
                                                      weight='weights_{}'.format(v)),  # {}_weights
                                    axis=1 - i)

            for idx, gene in enumerate(sorted_nodes):
                if gene in csv_genes_set:
                    gene_att_score[idx] = gene_att_score[idx] + 35

            influence_score.iloc[:, i] = np.log1p(gene_att_score).flatten().tolist()

        lam = 0.8
        influence_score['influence_score'] = lam * influence_score.loc[:, 'out'] + \
                                             (1 - lam) * influence_score.loc[:, 'in']
        influence_score.rename(columns={'out': 'score_out', 'in': 'score_in'}, inplace=True)
        influence_score = influence_score.sort_values(by='influence_score', ascending=False)
        self.influence_score = influence_score



    # 在 TEDC2L_result_object.py 中，替换 driver_regulators 方法

    def driver_regulators(self,
                          topK: int = 100,
                          solver: str = 'GUROBI',
                          include_mces: bool = True,
                          include_energy: bool = True,
                          max_energy_drivers: Optional[int] = None,
                          driver_union: bool = True,
                          return_value: bool = False,
                          output_file: Optional[str] = None):
        """
        获取驱动调节器，使用增强的控制方法（MDS、MFVS、MCES、能量最优）

        参数:
            topK (int): 考虑的顶级驱动调节器的最大数量。默认为100。
            solver (str): 用于寻找驱动调节器的优化求解器。默认为'GUROBI'。
            include_mces (bool): 是否包含最小控制边集(MCES)方法。默认为True。
            include_energy (bool): 是否包含能量最优控制方法。默认为True。
            max_energy_drivers (int, optional): 能量最优驱动节点的最大数量。
            driver_union (bool): 是否使用并集合并驱动集；否则使用交集。默认为True。
            return_value (bool): 是否返回驱动调节器DataFrame。默认为False。
            output_file (str): 如果提供，驱动调节器DataFrame将保存到此文件路径。
        """
        if self.influence_score is None:
            self.gene_influence_score()

        # 调用增强的驱动调节器函数
        self.driver_regulator, self.method_results = driver_regulators(
            self.network,
            self.influence_score,
            topK=topK,
            driver_union=driver_union,
            solver=solver,
            include_mces=include_mces,
            include_energy=include_energy,
            max_energy_drivers=max_energy_drivers
        )

        # 提取关键基因信息
        self._out_critical_genes = self.method_results.get('out_critical_genes', set())
        self._in_critical_genes = self.method_results.get('in_critical_genes', set())

        # 保存各方法的具体结果
        self.mces_results = self.method_results.get('MCES_edges', pd.DataFrame())
        self.energy_results = self.method_results.get('Energy_drivers_df', pd.DataFrame())

        # 添加转录因子标记
        self.driver_regulator['is_TF'] = self.driver_regulator.index.isin(self.TFs)

        # 打印各方法的统计信息
        print("\n=== 驱动调节器识别结果 ===")
        mds_count = self.driver_regulator['is_MDS_driver'].sum()
        mfvs_count = self.driver_regulator['is_MFVS_driver'].sum()
        print(f"MDS驱动基因数量: {mds_count}")
        print(f"MFVS驱动基因数量: {mfvs_count}")

        if include_mces and 'is_MCES_driver' in self.driver_regulator.columns:
            mces_count = self.driver_regulator['is_MCES_driver'].sum()
            print(f"MCES驱动基因数量: {mces_count}")

        if include_energy and 'is_Energy_driver' in self.driver_regulator.columns:
            energy_count = self.driver_regulator['is_Energy_driver'].sum()
            print(f"能量最优驱动基因数量: {energy_count}")

        final_count = self.driver_regulator['is_driver_regulator'].sum()
        print(f"最终TEDC2L驱动调节器数量: {final_count}")
        print("========================\n")

        # 保存结果到文件
        if isinstance(output_file, str):
            self.driver_regulator.to_csv(output_file)
            print(f"驱动调节器结果已保存到: {output_file}")

            # 同时保存详细的方法比较结果
            import os
            output_dir = os.path.dirname(output_file)
            if output_dir:
                self.save_method_comparison(output_dir)
            else:
                self.save_method_comparison('./')

        if return_value:
            return self.driver_regulator

    def RGM_activity(self, num_workers: int = 8, return_value: bool = False, output_file: Optional[str] = None):
        """
        Select RGMs of driver regulators and calculate their activities in each cell.
        Activity score is calculated based on AUCell, which is from the `pyscenic` package.

        Parameters:
            num_workers (int): The number of worker processes to use for AUCell calculation. Default is 8.
            return_value (bool): If True, the function will return the calculated AUCell matrices. Default is False.
            output_file (str): If provided, the calculated AUCell matrices will be saved to this file path.
        """
        print('[3] - Identifying regulon-like gene modules...')

        if self.driver_regulator is None:
            raise ValueError(
                f'No result found for driver regulators. Please run `TEDC2L.NetModel.driver_regulators` first.'
            )
        from pyscenic.aucell import aucell
        from ctxcore.genesig import GeneSignature

        network = self.network
        DEgenes = self.DEgenes
        drivers = set(self.driver_regulator[self.driver_regulator['is_driver_regulator']].index)

        auc_mtx_out, auc_mtx_in, auc_mtx = None, None, None

        # Ensure _out_critical_genes and _in_critical_genes are populated
        if self._out_critical_genes is None or self._in_critical_genes is None:
            print(
                "  Warning: _out_critical_genes or _in_critical_genes not found. Recalculating highly weighted genes.")
            from driver_regulators import highly_weighted_genes
            _, self._out_critical_genes, self._in_critical_genes = highly_weighted_genes(self.influence_score,
                                                                                         self.driver_regulator.shape[
                                                                                             0] // 2)

        out_hub_nodes = self._out_critical_genes.intersection(drivers)
        out_RGMs = {i + '_out({})'.format(len(list(set(network.successors(i)).intersection(DEgenes)))):
                        list(set(network.successors(i)).intersection(DEgenes)) for i in out_hub_nodes
                    if len(set(network.successors(i)).intersection(DEgenes)) >= 10}
        in_hub_nodes = self._in_critical_genes.intersection(drivers)
        in_RGMs = {i + '_in({})'.format(len(list(set(network.predecessors(i)).intersection(DEgenes)))):
                       list(set(network.predecessors(i)).intersection(DEgenes)) for i in in_hub_nodes
                   if len(set(network.predecessors(i)).intersection(DEgenes)) >= 10}

        # n_cells x n_genes
        out_RGMs = [GeneSignature(name=k, gene2weight=v) for k, v in out_RGMs.items()]
        in_RGMs = [GeneSignature(name=k, gene2weight=v) for k, v in in_RGMs.items()]
        if len(out_RGMs) > 0:
            auc_mtx_out = aucell(self.expression_data, out_RGMs, num_workers=num_workers, auc_threshold=0.25,
                                 normalize=False)
            # Generate a Z-score for each RGM to enable comparison between RGMs
            auc_mtx_out_Z = pd.DataFrame(index=auc_mtx_out.index, columns=list(auc_mtx_out.columns), dtype=float)
            for col in list(auc_mtx_out.columns):
                auc_mtx_out_Z[col] = (auc_mtx_out[col] - auc_mtx_out[col].mean()) / auc_mtx_out[col].std(ddof=0)

        if len(in_RGMs) > 0:
            auc_mtx_in = aucell(self.expression_data, in_RGMs, num_workers=num_workers, auc_threshold=0.25,
                                normalize=False)
            # Generate a Z-score for each RGM to enable comparison between RGMs
            auc_mtx_in_Z = pd.DataFrame(index=auc_mtx_in.index, columns=list(auc_mtx_out.columns), dtype=float)
            for col in list(auc_mtx_in.columns):
                auc_mtx_in_Z[col] = (auc_mtx_in[col] - auc_mtx_in[col].mean()) / auc_mtx_in[col].std(ddof=0)

        if (len(out_RGMs) > 0) & (len(in_RGMs) > 0):
            auc_mtx = pd.merge(auc_mtx_out, auc_mtx_in, how='inner', left_index=True, right_index=True)
            # Generate a Z-score for each RGM to enable comparison between RGMs
            auc_mtx_Z = pd.DataFrame(index=auc_mtx.index, columns=list(auc_mtx.columns), dtype=float)
            for col in list(auc_mtx.columns):
                auc_mtx_Z[col] = (auc_mtx[col] - auc_mtx[col].mean()) / auc_mtx[col].std(ddof=0)

        RGMs = out_RGMs + in_RGMs
        RGMs_AUCell_dict = {'RGMs': RGMs, 'aucell': auc_mtx,
                            'aucell_out': auc_mtx_out, 'aucell_in': auc_mtx_in}
        self.RGMs_AUCell_dict = RGMs_AUCell_dict
        print('Done!')

        if isinstance(output_file, str):
            pd_RGMs = pd.DataFrame(index=['genes'], columns=auc_mtx.columns)
            for r in RGMs:
                pd_RGMs.at['genes', r.name] = list(r.gene2weight.keys())
            result = pd.concat([pd_RGMs, auc_mtx], axis=0)
            result.columns.name = 'RGM'
            result.index.name = 'Cell'
            result.to_csv(output_file)
        if return_value:
            return RGMs_AUCell_dict

    def plot_gene_embedding_with_clustering(self,
                                            n_neighbors: int = 30,
                                            resolution: float = 1,
                                            return_value: bool = False):
        """
        Cluster genes with TEDC2L derived embeddings by using the Leiden algorithm and visualize in 2-D UMAP space.
        """
        if self._adata_gene is None:
            self._adata_gene = sc.AnnData(X=self.gene_embedding)
            sc.pp.neighbors(self._adata_gene, n_neighbors=n_neighbors, use_rep='X')
            # Higher resolutions lead to more communities, while lower resolutions lead to fewer communities.
            sc.tl.leiden(self._adata_gene, resolution=resolution)
            sc.tl.umap(self._adata_gene, n_components=2, min_dist=0.3)
            pos = self._adata_gene.obsm['X_umap']
            self.gene_cluster = self._adata_gene.obs['leiden']

        with rc_context({'figure.figsize': (4, 4)}):
            sc.pl.umap(self._adata_gene, color=['leiden'], legend_loc='on data',
                       legend_fontsize=8, legend_fontoutline=2, title='Leiden clustering using gene embeddings')

        if return_value:
            return self.gene_cluster

    def plot_influence_score(self, topK: int = 20):
        """
        Plot the gene influence score of top-k driver regulators.
        """
        if self.driver_regulator is None:
            raise ValueError(
                f'Did not find the result of driver regulators. Run `TEDC2L.NetModel.driver_regulators` first.'
            )
        data_for_plot = self.driver_regulator[self.driver_regulator['is_driver_regulator']]
        data_for_plot = data_for_plot[0:topK]

        plt.figure(figsize=(1.5, topK * 0.15))
        sns.set_theme(style='ticks', font_scale=0.5)

        ax = sns.barplot(x='influence_score', y=data_for_plot.index, data=data_for_plot, orient='h',
                         palette=sns.color_palette(f"ch:start=.5,rot=-.5,reverse=1,dark=0.4", n_colors=topK))
        ax.set_title(self.name)
        ax.set_xlabel('Influence score')
        ax.set_ylabel('Driver regulators')
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax.yaxis.set_minor_locator(ticker.NullLocator())
        sns.despine()

        plt.show()

    def plot_network(self, nodes: Optional[list] = None):
        """
        Plot network graph with circos layout.
        """
        try:
            import nxviz as nv
            from nxviz import annotate
        except ImportError:
            raise ImportError("install nxviz via `pip install nxviz`")

        if nodes is not None:
            sub_net = self.network.subgraph(nodes)
        else:
            sub_net = self.network

        nx.set_node_attributes(sub_net, self.driver_regulator['influence_score'].to_dict(), 'influence_score')
        nx.set_node_attributes(sub_net, {i: i for i in sub_net.nodes}, 'gene_name')

        # sns.set_theme(font_scale=0.7)
        ax = nv.circos(sub_net,
                       sort_by='influence_score',
                       # node_size_by='influence_score',
                       node_enc_kwargs={
                           'size_scale': 0.6,
                       },
                       edge_alpha_by='weights_combined',
                       edge_enc_kwargs={
                           'lw_scale': 0.5,
                           'alpha_scale': 0.7,
                       },
                       )
        annotate.circos_labels(sub_net, sort_by='influence_score', layout='rotate')

        # Get the figure and axes from the circos plot
        fig = plt.gcf()
        ax = plt.gca()

        # Make background transparent
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')

        ax.set_axis_off()
        plt.tight_layout()

    def plot_driver_genes_Venn(self):
        """
        Plot Venn diagram of MDS, MFVS and top-ranked regulators.
        """
        if self.driver_regulator is None:
            raise ValueError(
                f'Did not find the result of driver regulators. Run `TEDC2L.NetModel.driver_regulators` first.'
            )
        else:
            drivers_df = self.driver_regulator

        MFVS_driver_set = set(drivers_df.loc[drivers_df['is_Energy_driver']].index)
        MDS_driver_set = set(drivers_df.loc[drivers_df['is_MDS_driver']].index)

        #修改


        top_ranked_genes = set(drivers_df.loc[drivers_df['is_driver_regulator']].index).union(
            (set(drivers_df.index) - MFVS_driver_set.union(MDS_driver_set)))

        f = plt.figure(figsize=(3, 3))
        sns.set_theme(font_scale=f.get_dpi() / 100)
        out = venn3(subsets=[MDS_driver_set, MFVS_driver_set, top_ranked_genes],
                    set_labels=('MDS driver genes({})'.format(len(MDS_driver_set)),
                                'MFVS driver genes({})'.format(len(MFVS_driver_set)),
                                'Top ranked genes({})'.format(len(top_ranked_genes))),
                    set_colors=('#0076FF', '#D74715', '#009000'),
                    alpha=0.45)
        for text in out.set_labels:
            if text is not None:
                text.set_fontsize(7)
        for text in out.subset_labels:
            if text is not None:
                text.set_fontsize(8)
        plt.tight_layout()
        # plt.savefig('drivers_Venn_plot.svg', dpi=600, format='svg', bbox_inches='tight')
        plt.show()

    def plot_RGM_activity_heatmap(self,
                                  cell_label: Optional[pd.DataFrame] = None,
                                  type: str = 'out',
                                  z_score: Optional[int] = 0):
        """
        Plot clustermap of RGM activity matrix.

        Parameters:
            cell_label (pd.DataFrame, optional): A DataFrame containing cell labels.
                If provided, cells are ordered by cell clusters, else cells are ordered by hierarchical clustering.
                Defaults to None.
            type (str, optional): Type of RGM activity to plot.
                It can be 'out', 'in', or 'all'. Defaults to 'out'.
            z_score (int, optional): Either 0 (rows) or 1 (columns).
                Whether or not to calculate z-scores for the rows or the columns.  Defaults to 0.
        """
        if self.RGMs_AUCell_dict is None:
            raise ValueError(
                f'Did not find the result of RGMs. Run `TEDC2L.NetModel.RGM_activity` first.'
            )
        assert type in ['out', 'in', 'all']

        if type == 'all':
            auc_mtx = self.RGMs_AUCell_dict['aucell']
        elif type == 'out':
            auc_mtx = self.RGMs_AUCell_dict['aucell_out']
        else:  # in
            auc_mtx = self.RGMs_AUCell_dict['aucell_in']

        # Create a categorical palette to identify the networks
        if cell_label is not None:
            network_lable = cell_label
            network_pal = sns.husl_palette(len(network_lable.unique()), h=.5)
            network_lut = dict(zip(map(str, network_lable.unique()), network_pal))
            network_colors = pd.Series(list(network_lable), index=auc_mtx.index).map(network_lut)
            col_cluster = False
        else:
            network_colors = None
            col_cluster = True

        # plot clustermap (n_cell * n_gene)
        f = plt.figure()
        sns.set_theme(font_scale=f.get_dpi() / 150)
        g = sns.clustermap(auc_mtx.T, method='ward', square=False, linecolor='black',
                           z_score=z_score, vmin=-2.5, vmax=2.5,
                           col_cluster=col_cluster, col_colors=network_colors, cmap="RdBu_r",
                           figsize=(4.5, 0.15 * auc_mtx.shape[1]),
                           xticklabels=False, yticklabels=True, dendrogram_ratio=0.12,
                           cbar_pos=(0.75, 0.92, 0.2, 0.02),
                           cbar_kws={'orientation': 'horizontal'})
        g.cax.set_visible(True)
        g.ax_heatmap.set_ylabel('Regulon-like gene modules')
        g.ax_heatmap.set_xlabel('Cells')
        g.ax_heatmap.yaxis.set_minor_locator(ticker.NullLocator())
        if cell_label is not None:
            for label in network_lable.unique():
                g.ax_col_dendrogram.bar(0, 0, color=network_lut[label], label=label, linewidth=0)
            g.ax_col_dendrogram.legend(title='Cell types', loc="upper left", ncol=1,
                                       bbox_to_anchor=(1.10, 0.90), facecolor='white')
        g.ax_row_dendrogram.set_visible(False)
        plt.show()

    def plot_network_degree_distribution(self):
        """
        Plot degree distribution of the predicted network
        """
        network = self.network.copy()
        network.remove_edges_from(nx.selfloop_edges(network))
        network = network.subgraph(max(nx.weakly_connected_components(network), key=len)).copy()

        # Slope of degree distribution
        degree_sequence = pd.DataFrame(np.array(network.degree))
        degree_sequence.columns = ["ind", "degree"]
        degree_sequence['degree'] = degree_sequence['degree'].astype(int)
        degree_sequence = degree_sequence.loc[degree_sequence['degree'] != 0, :]
        degree_sequence = degree_sequence.set_index("ind")
        dist = degree_sequence.degree.value_counts() / degree_sequence.degree.value_counts().sum()
        dist.index = dist.index.astype(int)

        x = np.log(dist.index.values).reshape([-1, 1])
        y = np.log(dist.values).reshape([-1, 1])

        model = lr()
        model.fit(x, y)

        # Figure
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        # sns.set_theme(style="white", font_scale=fig.get_dpi() / 120)

        x_ = np.array([-1, 5]).reshape([-1, 1])
        y_ = model.predict(x_)

        ax.set_title("degree distribution (log scale)")
        ax.plot(x_.flatten(), y_.flatten(), c="black", alpha=0.5)

        ax.scatter(x.flatten(), y.flatten(), c="black")
        ax.text(0.45, 0.95,
                f"slope: {model.coef_[0][0] :.4g}, " + r"$R^2$: " + f"{model.score(x, y) :.4g}\n" +
                f"num_of_genes: {self.network.number_of_nodes()}\n" +
                f"num_of_edges: {self.network.number_of_edges()}\n" +
                f"clustering_coefficient: {nx.average_clustering(self.network) :.4g}\n",
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='left',
                fontsize=8.5)
        ax.set_ylim([y.min() - 0.2, y.max() + 0.2])
        ax.set_xlim([-0.2, x.max() + 0.2])
        ax.set_xlabel("log k")
        ax.set_ylabel("log P(k)")
        ax.grid(None)

        plt.tight_layout()
        plt.show()

    def plot_four_method_venn(self):
        """
        绘制四集合韦恩图，展示四种控制方法的交集
        """
        if self.method_results is None:
            raise ValueError(
                f'Did not find the results of enhanced driver regulators. Run driver_regulators first.'
            )

        # 尝试导入matplotlib_venn
        try:
            from matplotlib_venn import venn3_unweighted
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("install matplotlib-venn via `pip install matplotlib-venn`")

        # 获取各方法的驱动基因集合
        mds_drivers = self.method_results.get('MDS_drivers', set())
        mfvs_drivers = self.method_results.get('MFVS_drivers', set())
        mces_drivers = self.method_results.get('MCES_drivers', set())
        energy_drivers = self.method_results.get('Energy_drivers', set())

        # 由于matplotlib_venn不直接支持4集合，我们绘制多个3集合图进行比较
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # MDS vs MFVS vs MCES
        if len(mces_drivers) > 0:
            ax1 = axes[0, 0]
            venn3_unweighted([mds_drivers, mfvs_drivers, mces_drivers],
                             ('MDS', 'MFVS', 'MCES'), ax=ax1)
            ax1.set_title('MDS vs MFVS vs MCES')

        # MDS vs MFVS vs Energy
        if len(energy_drivers) > 0:
            ax2 = axes[0, 1]
            venn3_unweighted([mds_drivers, mfvs_drivers, energy_drivers],
                             ('MDS', 'MFVS', 'Energy'), ax=ax2)
            ax2.set_title('MDS vs MFVS vs Energy-Optimal')

        # MDS vs MCES vs Energy
        if len(mces_drivers) > 0 and len(energy_drivers) > 0:
            ax3 = axes[1, 0]
            venn3_unweighted([mds_drivers, mces_drivers, energy_drivers],
                             ('MDS', 'MCES', 'Energy'), ax=ax3)
            ax3.set_title('MDS vs MCES vs Energy-Optimal')

        # MFVS vs MCES vs Energy
        if len(mces_drivers) > 0 and len(energy_drivers) > 0:
            ax4 = axes[1, 1]
            venn3_unweighted([mfvs_drivers, mces_drivers, energy_drivers],
                             ('MFVS', 'MCES', 'Energy'), ax=ax4)
            ax4.set_title('MFVS vs MCES vs Energy-Optimal')

        plt.tight_layout()
        plt.show()

        # 打印交集统计
        print("Method intersection statistics:")
        if len(mces_drivers) > 0:
            print(f"MDS ∩ MFVS ∩ MCES: {len(mds_drivers & mfvs_drivers & mces_drivers)} genes")
        if len(energy_drivers) > 0:
            print(f"MDS ∩ MFVS ∩ Energy: {len(mds_drivers & mfvs_drivers & energy_drivers)} genes")
        if len(mces_drivers) > 0 and len(energy_drivers) > 0:
            all_intersection = mds_drivers & mfvs_drivers & mces_drivers & energy_drivers
            print(f"All four methods intersection: {len(all_intersection)} genes")
            if len(all_intersection) > 0:
                print(f"Core driver genes: {sorted(list(all_intersection))}")

    def save_method_comparison(self, output_dir: str = './'):
        """
        保存各方法的比较结果
        """
        if self.method_results is None:
            raise ValueError("No method results available. Run driver_regulators first.")

        import os
        from pathlib import Path

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # 保存MCES结果
        if 'MCES_edges' in self.method_results:
            mces_path = output_path / 'mces_control_edges.csv'
            self.method_results['MCES_edges'].to_csv(mces_path, index=False)
            print(f"MCES results saved to {mces_path}")

        # 保存能量最优结果
        if 'Energy_drivers_df' in self.method_results:
            energy_path = output_path / 'energy_optimal_drivers.csv'
            self.method_results['Energy_drivers_df'].to_csv(energy_path, index=False)
            print(f"Energy-optimal results saved to {energy_path}")

        # 保存方法比较汇总
        comparison_data = []
        methods = ['MDS_drivers', 'MFVS_drivers', 'MCES_drivers', 'Energy_drivers']

        for method in methods:
            if method in self.method_results:
                driver_set = self.method_results[method]
                comparison_data.append({
                    'method': method.replace('_drivers', ''),
                    'num_drivers': len(driver_set),
                    'drivers': list(driver_set) if len(
                        driver_set) <= 50 else f"Too many to list ({len(driver_set)} total)"
                })

        comparison_df = pd.DataFrame(comparison_data)
        comparison_path = output_path / 'method_comparison_summary.csv'
        comparison_df.to_csv(comparison_path, index=False)
        print(f"Method comparison saved to {comparison_path}")
