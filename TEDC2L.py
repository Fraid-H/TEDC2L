import argparse
from os import fspath
from pathlib import Path
import pandas as pd
import numpy as np

from cell_lineage_GRN import NetModel
from TEDC2L_result_object import TEDC2LResults
from utils import data_preparation

#需要修改gene.csv  以及 基因转录组表达文件

# TODO: 1) Modify the processing of input data to accommodate multiple lineages; 2) Improve the prompt message
# 同时修改 main() 函数中调用 driver_regulators 的部分（约第161行）
# 替换原来的调用为：

def main():
    parser = argparse.ArgumentParser(prog='TEDC2L', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = add_main_args(parser)
    args = parser.parse_args()

    ## output dir
    p = Path(args.out_dir)
    if not p.exists():
        Path.mkdir(p)

    # Load text embeddings if provided
    text_embeddings_df = None
    if args.text_emb_path:
        print(f"Loading text embeddings from {args.text_emb_path}")
        try:
            text_embeddings_df = pd.read_csv(args.text_emb_path, index_col=0, header=0)
            text_embeddings_df.index = text_embeddings_df.index.astype(str).str.upper()
            if args.text_emb_dim is not None:
                actual_dim = text_embeddings_df.shape[1]
                if actual_dim != args.text_emb_dim:
                    print(f"Warning: Expected text embedding dimension {args.text_emb_dim}, got {actual_dim}")
            print(
                f"Loaded text embeddings: {text_embeddings_df.shape[0]} genes × {text_embeddings_df.shape[1]} dimensions")
        except Exception as e:
            print(f"Error loading text embeddings: {e}")
            text_embeddings_df = None

    ## load data with enhanced parameters
    prep_kwargs = {
        'text_knn': args.text_knn,
        'text_sim_tau': args.text_sim_tau,
        'text_mix_alpha': args.text_mix_alpha,
        'text_max_edges': args.text_max_edges,
    }

    data = data_preparation(args.input_expData, args.input_priorNet, genes_DE=args.input_genesDE,
                            additional_edges_pct=args.additional_edges_pct, **prep_kwargs)

    # Add text embeddings to each lineage if available
    if text_embeddings_df is not None:
        for lineage_name, adata_l in data.items():
            genes = pd.Index(adata_l.var_names).astype(str).str.upper()
            aligned_embeddings = text_embeddings_df.reindex(genes).fillna(0.0).to_numpy().astype('float32')
            adata_l.varm['gene_text_emb'] = aligned_embeddings
            matched_genes = (text_embeddings_df.index.isin(genes)).sum()
            missing_genes = len(genes) - matched_genes
            print(
                f"  Lineage {lineage_name}: {matched_genes} genes matched, {missing_genes} genes missing from text embeddings")

    # For single lineage (backward compatibility)
    data = data['all']

    ## GRN construction with enhanced parameters
    TEDC2L_GRN_model = NetModel(hidden_dim=args.hidden_dim,
                                output_dim=args.output_dim,
                                heads=args.heads,
                                attention_type=args.attention,
                                miu=args.miu,
                                epochs=args.epochs,
                                repeats=args.repeats,
                                seed=args.seed,
                                cuda=args.cuda,
                                edge_gate_beta=args.edge_gate_beta,
                                loss_text_align=args.loss_text_align,
                                contrastive_proj_dim=args.contrastive_proj_dim,
                                contrastive_temperature=args.contrastive_temperature)

    # 记录实验配置
    config = {
        'text_emb_path': args.text_emb_path,
        'text_knn': args.text_knn,
        'text_sim_tau': args.text_sim_tau,
        'text_mix_alpha': args.text_mix_alpha,
        'edge_gate_beta': args.edge_gate_beta,
        'loss_text_align': args.loss_text_align,
        'contrastive_proj_dim': args.contrastive_proj_dim,
        'contrastive_temperature': args.contrastive_temperature,
        'include_mces': args.include_mces,
        'include_energy': args.include_energy,
        'max_energy_drivers': args.max_energy_drivers
    }
    print("实验配置:")
    for key, value in config.items():
        if value != 0 and value is not None and value != 1.0 and value != -1.0:
            print(f"  {key}: {value}")

    TEDC2L_GRN_model.run(data, showProgressBar=True)
    G_predicted = TEDC2L_GRN_model.get_network(keep_self_loops=~args.remove_self_loops,
                                               edge_threshold_avgDegree=args.edge_threshold_param,
                                               edge_threshold_zscore=None,
                                               output_file=fspath(p / 'cell_lineage_GRN.csv'))
    node_embeddings = TEDC2L_GRN_model.get_gene_embedding(output_file=fspath(p / 'gene_embs.csv'))
    TEDC2L_results = TEDC2LResults(adata=TEDC2L_GRN_model._adata,
                                   network=G_predicted,
                                   gene_embedding=node_embeddings)

    ## 基因影响分数
    TEDC2L_results.gene_influence_score()

    ## 增强的驱动调节器识别（包含四种方法）
    print("\n开始增强的驱动调节器识别...")
    TEDC2L_results.driver_regulators(
        topK=args.topK_drivers,
        include_mces=args.include_mces,
        include_energy=args.include_energy,
        max_energy_drivers=args.max_energy_drivers,
        output_file=fspath(p / 'driver_regulators.csv')
    )

    # 保存各种方法的详细结果
    if args.save_individual_results:
        print("\n保存各方法的详细结果...")

        # 保存MDS和MFVS结果
        mds_drivers = TEDC2L_results.method_results.get('MDS_drivers', set())
        mfvs_drivers = TEDC2L_results.method_results.get('MFVS_drivers', set())

        pd.DataFrame({'MDS_drivers': list(mds_drivers)}).to_csv(fspath(p / 'MDS_drivers.csv'), index=False)
        pd.DataFrame({'MFVS_drivers': list(mfvs_drivers)}).to_csv(fspath(p / 'MFVS_drivers.csv'), index=False)

        # 保存MCES结果
        if args.include_mces and not TEDC2L_results.mces_results.empty:
            TEDC2L_results.mces_results.to_csv(fspath(p / 'MCES_control_edges.csv'), index=False)
            mces_drivers = TEDC2L_results.method_results.get('MCES_drivers', set())
            pd.DataFrame({'MCES_drivers': list(mces_drivers)}).to_csv(fspath(p / 'MCES_drivers.csv'), index=False)

        # 保存能量最优结果
        if args.include_energy and not TEDC2L_results.energy_results.empty:
            TEDC2L_results.energy_results.to_csv(fspath(p / 'Energy_optimal_drivers.csv'), index=False)

    # 保存方法比较结果
    if args.save_method_comparison:
        TEDC2L_results.save_method_comparison(output_dir=args.out_dir)

    ## RGMs
    RGMs_results_dict = TEDC2L_results.RGM_activity(return_value=True)
    RGMs = pd.DataFrame(
        [{'Driver_Regulator': r.name, 'Members': list(r.gene2weight.keys())} for r in RGMs_results_dict['RGMs']])
    RGMs.to_csv(fspath(p / 'RGMs.csv'))
    RGMs_results_dict['aucell'].to_csv(fspath(p / 'AUCell_mtx.csv'))

    # 保存实验配置
    config_df = pd.DataFrame([config])
    config_df.to_csv(fspath(p / 'experimental_config.csv'), index=False)

    print(f'\n[完成!] 请检查结果在 "{args.out_dir}/" 目录中')
    print("\n生成的文件包括:")
    print("- driver_regulators.csv: 所有驱动调节器结果")
    print("- MDS_drivers.csv: MDS驱动基因")
    print("- MFVS_drivers.csv: MFVS驱动基因")
    if args.include_mces:
        print("- MCES_control_edges.csv: MCES控制边")
        print("- MCES_drivers.csv: MCES驱动基因")
    if args.include_energy:
        print("- Energy_optimal_drivers.csv: 能量最优驱动基因")
    print("- method_comparison_summary.csv: 方法比较总结")
    print("- experimental_config.csv: 实验配置")


def add_main_args(parser: argparse.ArgumentParser):
    # Input data
    input_parser = parser.add_argument_group(title='Input data options')

    input_parser.add_argument('--input_expData', type=str, default="../example_data/CRC_EMTAB8107_data_DE_C9.csv",
                              required=False, metavar='PATH',
                              help='path to the input gene expression data')

    # input_parser.add_argument('--input_priorNet', type=str, default="../prior_data/network_human.csv", required=False,
    #                           metavar='PATH',
    #                           help='path to the input prior gene interaction network')


    input_parser.add_argument('--input_priorNet', type=str, default="../prior_data/network-combined.csv", required=False,
                              metavar='PATH',
                              help='path to the input prior gene interaction network')



    input_parser.add_argument('--input_genesDE', type=str, default=None, required=False, metavar='PATH',
                              help='path to the input gene differential expression score')
    input_parser.add_argument('--additional_edges_pct', type=float, default=0.01,
                              help='proportion of high co-expression interactions to be added')

    # Enhanced control methods (修改这部分，确保默认启用所有方法)
    enhanced_parser = parser.add_argument_group(title='Enhanced control method options')
    enhanced_parser.add_argument('--include_mds', action='store_true', default=True,
                                 help='include MDS (Minimum Dominating Set) method')
    enhanced_parser.add_argument('--include_mfvs', action='store_true', default=True,
                                 help='include MFVS (Minimum Feedback Vertex Set) method')
    enhanced_parser.add_argument('--include_mces', action='store_true', default=True,
                                 help='include MCES (Minimum Control Edge Set) method')
    enhanced_parser.add_argument('--include_energy', action='store_true', default=True,
                                 help='include Energy-Optimal control method')
    enhanced_parser.add_argument('--max_energy_drivers', type=int, default=None,
                                 help='maximum number of energy-optimal driver nodes')
    enhanced_parser.add_argument('--save_method_comparison', action='store_true', default=True,
                                 help='save detailed comparison of all control methods')
    enhanced_parser.add_argument('--save_individual_results', action='store_true', default=True,
                                 help='save individual results for each method')

    # Text embedding options
    text_parser = parser.add_argument_group(title='Text embedding enhancement options')
    text_parser.add_argument('--text_emb_path', type=str, default="../example_data/gene_embs_CRC_EMTAB8107_data_DE_C9.csv", metavar='PATH',
                             help='path to gene text embeddings CSV file (gene names as index)')
    text_parser.add_argument('--text_emb_dim', type=int, default=1536,
                             help='expected dimension of text embeddings (optional validation)')
    text_parser.add_argument('--text_knn', type=int, default=10,
                             help='number of top-k text similarity neighbors to add as edges (0=disabled)')
    text_parser.add_argument('--text_sim_tau', type=float, default=0.0,
                             help='text similarity threshold for edge addition (0=disabled, only used when text_knn=0)')
    text_parser.add_argument('--text_mix_alpha', type=float, default=0.3,
                             help='mixing weight for combined Spearman+text similarity (-1=union strategy, >=0=mixed ranking)')
    text_parser.add_argument('--text_max_edges', type=int, default=0,
                             help='maximum number of text similarity edges to add (0=unlimited)')

    # GRN construction options
    grn_parser = parser.add_argument_group(title='Cell-lineage-specific GRN construction options')
    grn_parser.add_argument('--cuda', type=int, default=0,
                            help="an integer greater than -1 indicates the GPU device number and -1 indicates the CPU device")
    grn_parser.add_argument('--seed', type=int, default=2023,
                            help="random seed (set to -1 means no random seed is assigned)")
    grn_parser.add_argument("--hidden_dim", type=int, default=128,
                            help="hidden dimension of the GNN encoder")
    grn_parser.add_argument("--output_dim", type=int, default=64,
                            help="output dimension of the GNN encoder")
    grn_parser.add_argument("--heads", type=int, default=4,
                            help="number of heads of multi-head attention. Default is 4")
    grn_parser.add_argument("--attention", type=str, default='COS', choices=['COS', 'AD', 'SD'],
                            help="type of attention scoring function ('COS', 'AD', 'SD'). Default is 'COS'")
    grn_parser.add_argument('--miu', type=float, default=0.5,
                            help='parameter for considering the importance of attention coefficients of the first GNN layer')
    grn_parser.add_argument('--epochs', type=int, default=500,
                            help='number of epochs for one run')
    grn_parser.add_argument('--repeats', type=int, default=1,
                            help='number of run repeats')
    grn_parser.add_argument("--edge_threshold_param", type=int, default=8,
                            help="threshold for selecting top-weighted edges (larger values means more edges)")
    grn_parser.add_argument("--remove_self_loops", action="store_true",
                            help="remove self loops")

    # Enhanced GRN options
    enhanced_grn_parser = parser.add_argument_group(title='Enhanced GRN options')
    enhanced_grn_parser.add_argument('--edge_gate_beta', type=float, default=0.3,
                                     help='edge-level gating parameter (1.0=node-level only, 0.0=text-similarity only)')
    enhanced_grn_parser.add_argument('--loss_text_align', type=float, default=0.1,
                                     help='weight for graph-text contrastive learning loss (0.0=disabled)')
    enhanced_grn_parser.add_argument('--contrastive_proj_dim', type=int, default=128,
                                     help='projection dimension for contrastive learning')
    enhanced_grn_parser.add_argument('--contrastive_temperature', type=float, default=0.07,
                                     help='temperature parameter for contrastive learning')

    # Driver regulators
    driver_parser = parser.add_argument_group(title='Driver regulator identification options')
    driver_parser.add_argument('--topK_drivers', type=int, default=100,
                               help="number of top-ranked candidate driver genes according to their influence scores")

    # Output dir
    parser.add_argument("--out_dir", type=str, required=False, default='../output',
                        help="results output path")

    return parser


if __name__ == "__main__":
    main()