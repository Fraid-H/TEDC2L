import matplotlib

# 修正：在所有 matplotlib 操作之前，设置后端为 'Agg' (非交互模式，适合服务器)
matplotlib.use('Agg')

import pandas as pd
import networkx as nx
import scanpy as sc
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import numpy as np
import warnings

# --- 新增导入：用于绘制韦恩图 ---
from matplotlib_venn import venn2, venn3
from matplotlib import rcParams

# 导入TEDC2L的结果类 (假设此文件在当前目录下)
from TEDC2L_result_object import TEDC2LResults

# ================= 风格配置 (来自 julei.py) =================
# 忽略警告
warnings.filterwarnings("ignore")

# 字体与字号设置 (Nature Style)
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']
rcParams['font.size'] = 10
rcParams['axes.titlesize'] = 12
rcParams['axes.labelsize'] = 11
rcParams['legend.fontsize'] = 9
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['savefig.dpi'] = 300
rcParams['figure.dpi'] = 300

# 自定义配色方案
COLORS = {
    'MFVS': '#E64B35',  # 红色/橙红色
    'Energy': '#8491B4',  # 灰紫色/蓝紫色
    'TopRank': '#00A087',  # 蓝绿色
    'Both': '#3C5488'  # 深蓝色
}


# ==========================================================

def reconstruct_results_from_files(output_dir: Path):
    """
    从输出文件中加载数据并重建一个 TEDC2LResults 对象用于可视化。
    """
    print(f"--- Loading results from '{output_dir}' ---")

    # 1. 加载网络
    network_file = output_dir / 'cell_lineage_GRN.csv'
    if not network_file.exists():
        raise FileNotFoundError(f"Network file not found: {network_file}")

    edge_df = pd.read_csv(network_file, header=None, names=['source', 'target', 'weights_combined'])
    network = nx.from_pandas_edgelist(
        edge_df, source='source', target='target', edge_attr=['weights_combined'], create_using=nx.DiGraph
    )
    print(f"Loaded network with {network.number_of_nodes()} nodes and {network.number_of_edges()} edges.")

    # 2. 加载基因嵌入
    embedding_file = output_dir / 'gene_embs.csv'
    gene_embedding = pd.read_csv(embedding_file, index_col='geneName')

    # 3. 加载驱动调控因子
    driver_file = output_dir / 'driver_regulators.csv'
    driver_regulator = pd.read_csv(driver_file, index_col=0)

    # 4. 加载RGM活性矩阵
    aucell_file = output_dir / 'AUCell_mtx.csv'
    aucell_mtx = pd.read_csv(aucell_file, index_col=0)

    # 5. 创建 TEDC2LResults 对象
    dummy_adata = sc.AnnData(pd.DataFrame(index=['cell1'], columns=list(network.nodes())))
    dummy_adata.uns['name'] = 'all'
    if 'is_TF' in driver_regulator.columns:
        dummy_adata.var['is_TF'] = driver_regulator['is_TF']

    results = TEDC2LResults(adata=dummy_adata, network=network, gene_embedding=gene_embedding)

    results.driver_regulator = driver_regulator
    results.influence_score = driver_regulator
    results.RGMs_AUCell_dict = {'aucell': aucell_mtx, 'aucell_out': aucell_mtx, 'aucell_in': aucell_mtx}

    topK = 50
    v_out = driver_regulator.sort_values(by='score_out', ascending=False).head(topK)
    results._out_critical_genes = set(v_out[v_out['score_out'] > 0].index)

    v_in = driver_regulator.sort_values(by='score_in', ascending=False).head(topK)
    results._in_critical_genes = set(v_in[v_in['score_in'] > 0].index)

    print("--- TEDC2LnResults object reconstructed successfully ---\n")
    return results


def plot_custom_venn(drivers_df, save_path, top_k=50):
    """
    使用 julei.py 中的逻辑绘制韦恩图：
    比较 MFVS, Energy-Optimal (如果存在) 和 Top-Ranked Genes
    """
    # 1. 提取集合
    # 提取 MFVS 基因
    mfvs_genes = set(drivers_df[drivers_df.get('is_MFVS_driver', False) == True].index)

    # 检查是否有 Energy 结果
    has_energy = 'is_Energy_driver' in drivers_df.columns
    if has_energy:
        energy_genes = set(drivers_df[drivers_df['is_Energy_driver'] == True].index)
    else:
        energy_genes = set()

    # Influence Score Top K
    # 优先使用 'influence_score' 列，如果不存在则尝试使用 'score_out' 或其他逻辑，这里假设 'influence_score' 存在
    score_col = 'influence_score' if 'influence_score' in drivers_df.columns else 'score_out'
    top_genes = set(drivers_df.sort_values(score_col, ascending=False).head(top_k).index)

    plt.figure(figsize=(5, 5))

    # 2. 绘图
    if not has_energy or len(energy_genes) == 0:
        # 仅比较 MFVS 和 Top Rank (Venn2)
        venn2([mfvs_genes, top_genes],
              set_labels=('MFVS', f'Top-{top_k} Rank'),
              set_colors=(COLORS['MFVS'], COLORS['TopRank']),
              alpha=0.6)
    else:
        # 比较三者 (Venn3)
        v = venn3([mfvs_genes, energy_genes, top_genes],
                  set_labels=('MFVS', 'Energy-Opt', f'Top-{top_k} Rank'),
                  set_colors=(COLORS['MFVS'], COLORS['Energy'], COLORS['TopRank']),
                  alpha=0.6)

        # 优化标签显示字体
        try:
            for text in v.set_labels:
                if text: text.set_fontsize(10)
        except:
            pass

    plt.title("Overlap of Control Methods & Top Ranked Genes", pad=15)
    plt.tight_layout()
    plt.savefig(save_path, transparent=True, bbox_inches='tight')
    plt.close()
    print(f"Saved custom Venn diagram to {save_path}")


if __name__ == "__main__":
    # 指定你的输出文件夹路径
    output_directory = Path('../output__EMTAB8107_data_DE_C8')

    # 创建一个文件夹来存放所有图片
    plots_dir = output_directory / 'plots'
    plots_dir.mkdir(exist_ok=True)

    # 重建结果对象
    TEDC2L_results = reconstruct_results_from_files(output_directory)

    # --- 可视化并保存文件 ---

    print("1. Plotting and saving Venn diagram of driver genes (New Style)...")
    # 使用新的函数替代 TEDC2L_results.plot_driver_genes_Venn()
    plot_custom_venn(
        TEDC2L_results.driver_regulator,
        save_path=plots_dir / "1_venn_diagram.png",
        top_k=50
    )

    print("\n2. Plotting and saving influence score of top 20 driver regulators...")
    TEDC2L_results.plot_influence_score(topK=20)
    plt.suptitle("Influence Score of Top 20 Drivers")
    plt.savefig(plots_dir / "2_influence_score.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("\n3. Plotting and saving RGM activity heatmap...")
    # 注意：如果 julei.py 的 heatmap 更好看，你也可以把那个函数移植过来替换这里
    # 这里暂时保留原版的 heatmap
    TEDC2L_results.plot_RGM_activity_heatmap(type='all', z_score=0)
    plt.suptitle("RGM Activity Heatmap")
    plt.savefig(plots_dir / "3_rgm_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("\n4. Plotting and saving gene embedding with Leiden clustering (UMAP)...")
    TEDC2L_results.plot_gene_embedding_with_clustering()
    plt.savefig(plots_dir / "4_gene_embedding_umap.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("\n5. Plotting and saving network degree distribution...")
    TEDC2L_results.plot_network_degree_distribution()
    plt.savefig(plots_dir / "5_degree_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n--- Visualization complete! All plots are saved in the '{plots_dir}' folder. ---")