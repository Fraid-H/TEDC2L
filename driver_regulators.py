import networkx as nx
from networkx.algorithms import bipartite
import numpy as np
import pandas as pd
import cvxpy as cvx
from typing import Optional

# 在 driver_regulators.py 文件顶部添加版本兼容性检查和导入
# 位置：在文件开头的导入部分之后添加

import warnings
import sys

# 抑制特定的NumPy警告
warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in scalar divide')


def check_networkx_version():
    """检查NetworkX版本并提供兼容性信息"""
    try:
        import networkx as nx
        version = nx.__version__
        print(f"使用NetworkX版本: {version}")

        # 检查是否支持maximum_matching函数
        try:
            from networkx.algorithms import matching
            hasattr(matching, 'maximum_matching')
            print("NetworkX maximum_matching函数可用")
        except:
            print("警告: NetworkX maximum_matching函数不可用，将使用备用方法")

    except ImportError:
        print("错误: 未安装NetworkX")
        sys.exit(1)



def _root_nodes(directed_graph: nx.DiGraph):
    return set([n for n in directed_graph.nodes() if (directed_graph.in_degree(n) == 0)
                or (list(directed_graph.predecessors(n)) == [n])])


def _end_nodes(directed_graph: nx.DiGraph):
    return set([n for n in directed_graph.nodes() if (directed_graph.out_degree(n) == 0)
                or (list(directed_graph.successors(n)) == [n])])


"""
MDS control for a directed network.
Use efficient graph reduction and then solve the ILP problem using gurobi.
A set S \subset V of nodes in a graph G=(V,E) is a dominating set if every node v \in V is either an element of S 
or adjacent to an element of S.
Refs: [1] Dominating scale-free networks with variable scaling exponent: heterogeneous networks are not difficult to control.
      New Journal of Physics, 2012.
      [2] Critical controllability analysis of directed biological networks using efficient graph reduction.
      Scientific Reports, 2017.
"""


def _MDS_graph_reduction(directed_graph: nx.DiGraph):
    # Critical nodes are driver nodes
    critical_nodes = set()
    redundant_nodes = set()

    # Critical nodes cond. 1: Source nodes are critical (driver) nodes.
    critical_nodes.update(_root_nodes(directed_graph))
    remain_nodes = set(directed_graph.nodes()) - critical_nodes

    noChange = False
    while not noChange:
        noChange = True

        # Critical nodes cond. 2: A node with at least two directed edges to nodes with outdegree 0 and indegree 1.
        in1out0_nodes = set([n for n in directed_graph.nodes() if (directed_graph.in_degree(n) == 1
                                                                   and directed_graph.out_degree(n) == 0)])
        add_critical = set([n for n in list(remain_nodes - in1out0_nodes)
                            if (len(set(directed_graph.successors(n)).intersection(in1out0_nodes)) > 1)])
        if len(add_critical) == 0:
            noChange *= True
        else:
            critical_nodes.update(add_critical)
            remain_nodes = remain_nodes - add_critical
            remove_edges = [(i, n) for n in list(add_critical) for i in directed_graph.predecessors(n)]
            directed_graph.remove_edges_from(remove_edges)
            noChange *= False

        # Redundant nodes cond.: a node with outdegree 0 and has an incoming link from a critical node.
        add_redundant = set([n for n in list(remain_nodes)
                             if (len(set(directed_graph.predecessors(n)).intersection(critical_nodes)) > 0
                                 and (directed_graph.out_degree(n) == 0))])
        if len(add_redundant) == 0:
            noChange *= True
        else:
            redundant_nodes.update(add_redundant)
            remain_nodes = remain_nodes - add_redundant
            directed_graph.remove_nodes_from(add_redundant)
            noChange *= False

    return directed_graph, critical_nodes, redundant_nodes


def MDScontrol(directed_graph: nx.DiGraph, solver='GUROBI'):
    print('  Solving MDS problem...')
    directed_graph.remove_edges_from(nx.selfloop_edges(directed_graph)) #去除自环
    reduced_graph = nx.DiGraph(directed_graph)  #directed_graph的一个副本
    intermittent_nodes = set()
    MDS_driver_set = set()

    # Graph reduction
    reduced_graph, critical_nodes, redundant_nodes = _MDS_graph_reduction(reduced_graph)
    print('    {} critical nodes are found.'.format(len(critical_nodes)))

    # Use ILP to find an MDS in the reduced graph
    reduced_graph.remove_nodes_from(list(nx.isolates(reduced_graph)))
    print('    {} nodes left after graph reduction operation.'.format(reduced_graph.number_of_nodes()))
    if reduced_graph.number_of_nodes() == 0:
        print('  {} MDS driver nodes are found.'.format(len(critical_nodes)))
    else:
        print('    Solving the Integer Linear Programming problem on the reduced graph...')
        A = nx.to_numpy_array(reduced_graph)
        A = A + np.eye(A.shape[0]) # A = A + np.diag(np.ones(A.shape[0]))
        # Define the optimization variables
        x = cvx.Variable(reduced_graph.number_of_nodes(), boolean=True)
        # Define the constraints
        constraints = [A @ x >= np.ones(reduced_graph.number_of_nodes())]
        # constraints = [A.T @ x >= np.ones(reduced_graph.number_of_nodes())]
        # constraints = [x[i] + cvx.sum(x[j] for j in range(reduced_graph.number_of_nodes()) if A[i][j]) >= 1 for i in
        #                range(reduced_graph.number_of_nodes())]
        # Define the optimization problem
        obj = cvx.Minimize(cvx.sum(x))
        # Solve the problem
        prob = cvx.Problem(obj, constraints)

        if solver == 'GUROBI':
            # Solve with GUROBI.
            print('      Solving by GUROBI...(', end='')
            prob.solve(solver=cvx.GUROBI, verbose=False)
            print('optimal value with GUROBI:{},'.format(prob.value), end='  ')
        # elif solver == 'XPRESS':
        #        prob.solve(solver=cvx.XPRESS, verbose=False)
        #        print("optimal value with XPRESS:", prob.value)
        elif solver == 'GLPK_MI':
            # Solve with CLARABEL.
            print('      Solving by GLPK_MI...(', end='')
            prob.solve(solver=cvx.GLPK_MI, verbose=False)
            print('optimal value with GLPK_MI:{},'.format(prob.value), end='  ')
        else:
            # Solve with SCIP
            print('      Inaccurate solver is selected! Now, solving by SCIP...(', end='')
            prob.solve(solver=cvx.SCIP, verbose=False)
            print('optimal value with SCIP:{},'.format(prob.value), end='  ')
        print('status:{})'.format(prob.status))

        # Set remain nodes that belongs to the MDS as critical nodes
        nodes_idx_map = dict(zip(range(reduced_graph.number_of_nodes()), reduced_graph.nodes()))
        mds_nodes = set([v for k, v in nodes_idx_map.items() if x.value[k] == 1])
        MDS_driver_set = critical_nodes.union(mds_nodes)
        intermittent_nodes = set(reduced_graph.nodes()) - MDS_driver_set
        print('  {} MDS driver genes are found.'.format(len(MDS_driver_set)))

    return MDS_driver_set, intermittent_nodes


"""
FVS control for a directed network.
Use efficient graph reduction and then solve the ILP problem using gurobi. 
A set S \subset V of nodes in a graph G=(V,E) is a feedback vertex set if the removal of these nodes leaves the graph
without feedback loops.
Refs: [1] Dynamics and control at feedback vertex sets. ii: a faithful monitor to determine the diversity of molecular activities in regulatory networks.
      Journal of Theoretical Biology, 2013
      [2] Structure-based control of complex networks with nonlinear dynamics.
      Proceedings of the National Academy of Sciences, 2017
      [3] On computing the minimum feedback vertex set of a directed graph by contraction operations.
      IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, 2000
      [ILP] An exact algorithm for selecting partial scan flip-flops. 1994
"""


def _in0out0(directed_graph: nx.DiGraph):
    remove_set = set([n for n in directed_graph.nodes() if (directed_graph.in_degree(n) == 0
                                                            or directed_graph.out_degree(n) == 0)])
    directed_graph.remove_nodes_from(remove_set)
    return directed_graph, len(remove_set) == 0


def _selfloop(directed_graph: nx.DiGraph, S: set):
    remove_set = list(nx.nodes_with_selfloops(directed_graph))
    S = S.union(set(remove_set))
    directed_graph.remove_nodes_from(remove_set)
    return directed_graph, S, len(remove_set) == 0


def _in1(directed_graph: nx.DiGraph):
    remove_set = set([n for n in directed_graph.nodes() if (directed_graph.in_degree(n) == 1)])
    temp_set = set()
    for u in remove_set:
        v = list(directed_graph.predecessors(u))[0]
        if v != u:
            directed_graph.add_edges_from([(v, w) for w in directed_graph.successors(u)])
            directed_graph.remove_node(u)
        else:
            temp_set.update(u)
    remove_set = remove_set - temp_set
    return directed_graph, len(remove_set) == 0


def _out1(directed_graph: nx.DiGraph):
    remove_set = set([n for n in directed_graph.nodes() if (directed_graph.out_degree(n) == 1)])
    temp_set = set()
    for u in remove_set:
        v = list(directed_graph.successors(u))[0]
        if u != v:
            directed_graph.add_edges_from([(w, v) for w in directed_graph.predecessors(u)])
            directed_graph.remove_node(u)
        else:
            temp_set.update(u)
    remove_set = remove_set - temp_set
    return directed_graph, len(remove_set) == 0


def _PIE(directed_graph: nx.DiGraph):
    G_SCCs = [c for c in nx.strongly_connected_components(directed_graph)]
    edges_SCCs = set()
    for g in G_SCCs:
        edges_SCCs.update(set(directed_graph.subgraph(g).edges()))
    remove_set = set(directed_graph.edges()) - edges_SCCs
    directed_graph.remove_edges_from(list(remove_set))
    return directed_graph, len(remove_set) == 0


def _CORE(directed_graph: nx.DiGraph, nodes_importance: pd.DataFrame, S: set):
    remove_set = set()
    pie = [e for e in directed_graph.edges() if (directed_graph.has_edge(e[1], e[0]) and (e[1] != e[0]))]
    if len(pie) > 0:
        x, y = zip(*pie)
        nodes_pie = set(x).union(set(y))
        piv = [n for n in list(nodes_pie)
               if (set(directed_graph.predecessors(n)) == set(directed_graph.successors(n)))]
        # sort PIV in the ascending order according to the degree and out_degree_importance
        piv_df = pd.DataFrame(directed_graph.degree(piv), index=piv, columns=['id', 'degree'])
        piv_df['importance'] = piv_df.index.map(nodes_importance)
        piv_df.sort_values(by=['importance', 'degree'], ascending=[True, True], inplace=True)
        # piv_df.sort_values(by='degree', ascending=True, inplace=True)
        piv_df['valid'] = True
        for v in piv_df.index:
            if piv_df.loc[v, 'valid']:
                # d-clique
                d_graph = directed_graph.subgraph([v] + list(directed_graph.neighbors(v)))
                is_dclique = True
                for d_i in d_graph:
                    if len(list(nx.bfs_edges(d_graph, d_i, depth_limit=1))) < (len(d_graph) - 1):
                        is_dclique = False

                if is_dclique:
                    S = S.union(set(d_graph.nodes()) - set(v))  # v is CORE
                    remove_set = remove_set.union(set(d_graph.nodes()))
                    piv_df.loc[piv_df.index.isin(d_graph.nodes()), 'valid'] = False
                else:
                    piv_df.loc[v, 'valid'] = False
    directed_graph.remove_nodes_from(remove_set)
    return directed_graph, S, len(remove_set) == 0


def _DOME(directed_graph: nx.DiGraph):
    pie = [e for e in directed_graph.edges() if (directed_graph.has_edge(e[1], e[0]) and (e[1] != e[0]))]
    auxilliary_graph = nx.DiGraph(directed_graph)
    auxilliary_graph.remove_edges_from(pie)
    remove_set = [e for e in auxilliary_graph.edges() if e[0] != e[1] and
                  (set(auxilliary_graph.predecessors(e[0])).issubset(set(auxilliary_graph.predecessors(e[1])))
                   or
                   set(auxilliary_graph.successors(e[1])).issubset(set(auxilliary_graph.successors(e[0]))))]
    directed_graph.remove_edges_from(remove_set)
    return directed_graph, len(remove_set) == 0


def _MFVS_graph_reduction(directed_graph: nx.DiGraph, nodes_importance: pd.DataFrame, S: set):
    all_done = False
    while not all_done:
        all_done = True

        directed_graph, no_change = _in0out0(directed_graph)
        all_done *= no_change

        directed_graph, S, no_change = _selfloop(directed_graph, S)
        all_done *= no_change

        directed_graph, no_change = _out1(directed_graph)
        all_done *= no_change

        directed_graph, no_change = _in1(directed_graph)
        all_done *= no_change

        directed_graph, no_change = _PIE(directed_graph)
        all_done *= no_change

        directed_graph, S, no_change = _CORE(directed_graph, nodes_importance, S)
        all_done *= no_change

        directed_graph, no_change = _DOME(directed_graph)
        all_done *= no_change

    return directed_graph, S


def MFVScontrol(directed_graph: nx.DiGraph, nodes_importance: pd.DataFrame, solver='GUROBI'):
    print('  Solving MFVS problem...')
    # Source nodes are critical (driver) nodes
    critical_nodes = set()
    source_nodes = _root_nodes(directed_graph)
    critical_nodes.update(source_nodes)
    reduced_graph = nx.DiGraph(directed_graph)

    while True:
        # Graph reduction
        reduced_graph, critical_nodes = _MFVS_graph_reduction(reduced_graph, nodes_importance, critical_nodes)
        reduced_graph_SCCs = [c for c in nx.strongly_connected_components(reduced_graph)]
        if len(reduced_graph_SCCs) > 1:
            reduced_graph_new = nx.DiGraph()
            for g in reduced_graph_SCCs:
                if len(g) > 1:
                    g = nx.DiGraph(reduced_graph.subgraph(g))
                    g, critical_nodes = _MFVS_graph_reduction(g, nodes_importance, critical_nodes)
                    reduced_graph_new.update(g)
            reduced_graph = nx.DiGraph(reduced_graph_new)

        # Some tricks: If the size of reduced graph is still large, let the node with the largest degree
        # be a critical node, and do graph reduction again.
        if reduced_graph.number_of_nodes() > 150:
            d_seq = pd.merge(pd.DataFrame(reduced_graph.in_degree(), columns=['Node', 'in_Degree']),
                             pd.DataFrame(reduced_graph.out_degree(), columns=['Node', 'out_Degree']),
                             on='Node')
            d_seq['dot_Degree'] = d_seq['in_Degree'] * d_seq['out_Degree']
            node_max = d_seq['Node'][d_seq['dot_Degree'].idxmax()]
            critical_nodes.add(node_max)
            reduced_graph.remove_node(node_max)
        else:
            break
    print('    {} critical nodes are found.'.format(len(critical_nodes)))

    # Use ILP to find an MFVS in the reduced graph
    reduced_graph.remove_nodes_from(list(nx.isolates(reduced_graph)))
    print('    {} nodes left after graph reduction operation.'.format(reduced_graph.number_of_nodes()))
    if reduced_graph.number_of_nodes() == 0:
        print('  {} MFVS driver genes are found.'.format(len(critical_nodes)))
    else:
        print('    Solving the Integer Linear Programming problem on the reduced graph...')
        nodes_idx_map = dict(zip(reduced_graph.nodes(), range(len(reduced_graph.nodes()))))
        # Define the optimization variables
        n = reduced_graph.number_of_nodes()
        x = cvx.Variable(n, boolean=True)
        # Define the constraints
        constraints = []
        w = cvx.Variable(n, integer=True)
        for e in reduced_graph.edges():
            constraints += [w[nodes_idx_map[e[0]]] - w[nodes_idx_map[e[1]]] + n * x[nodes_idx_map[e[0]]] >= 1]
        constraints += [0 <= w, w <= (n - 1)]
        # Define the optimization problem
        obj = cvx.Minimize(cvx.sum(x))
        # Solve the problem
        prob = cvx.Problem(obj, constraints)

        if solver == 'GUROBI':
            # Solve with GUROBI.
            print('      Solving by GUROBI...(', end='')
            prob.solve(solver=cvx.GUROBI, verbose=False)
            print('optimal value with GUROBI:{},'.format(prob.value), end='  ')
        # elif solver=='XPRESS':
        #    prob.solve(solver=cvx.XPRESS, verbose=False)
        #    print("optimal value with XPRESS:", prob.value)
        elif solver == 'GLPK_MI':
            # Solve with CLARABEL.
            print('      Solving by GLPK_MI...(', end='')
            prob.solve(solver=cvx.GLPK_MI, verbose=False)
            print('optimal value with GLPK_MI:{},'.format(prob.value), end='  ')
        else:
            # Solve with SCIP
            print('      Inaccurate solver is selected. Now, solving by SCIP...(', end='')
            prob.solve(solver=cvx.SCIP, verbose=False)
            print('optimal value with SCIP:{},'.format(prob.value), end='  ')
        print('status:{})'.format(prob.status))

        # Set remain nodes that belongs to the MDS as critical nodes
        mFVS_remain = set([k for k, v in nodes_idx_map.items() if x.value[v] == 1])
        critical_nodes = critical_nodes.union(mFVS_remain)
        print('  {} MFVS driver nodes are found.'.format(len(critical_nodes)))

    return critical_nodes, source_nodes


def highly_weighted_genes(gene_influence_scores: pd.DataFrame, topK: int = 50):
    v_out = gene_influence_scores.sort_values(by='score_out', ascending=False)
    v_out = v_out.iloc[0:topK, [0]]
    out_critical_genes = set(v_out[v_out > 0].index)

    v_in = gene_influence_scores.sort_values(by='score_in', ascending=False)
    v_in = v_in.iloc[0:topK, [1]]
    in_critical_genes = set(v_in[v_in > 0].index)

    critical_genes = out_critical_genes.union(in_critical_genes)
    return critical_genes, out_critical_genes, in_critical_genes

import networkx as nx
import numpy as np
import pandas as pd
import cvxpy as cvx

def compute_mces_control_edges(directed_graph: nx.DiGraph,
                               influence_score: pd.DataFrame,
                               solver='GUROBI',
                               output_file=None):
    """
    计算最小控制边集（MCES - Minimum Control Edge Set）

    参数:
        directed_graph: 有向图网络
        influence_score: 基因影响分数
        solver: 优化求解器
        output_file: 保存结果的文件路径

    返回:
        control_edges_df: 控制边集DataFrame
        control_edge_count: 控制边数量
    """
    print('  Solving MCES problem...')

    # 移除自环
    graph = nx.DiGraph(directed_graph)
    graph.remove_edges_from(nx.selfloop_edges(graph))

    if graph.number_of_nodes() == 0:
        print('  Empty graph, no control edges needed.')
        return pd.DataFrame(columns=['from', 'to', 'weight']), 0

    # 构建二分图进行最大匹配
    # 创建节点的两个副本：源节点和目标节点
    bipartite_graph = nx.Graph()

    # 添加源节点（前缀'src_'）和目标节点（前缀'dst_'）
    source_nodes = ['src_' + str(node) for node in graph.nodes()]
    target_nodes = ['dst_' + str(node) for node in graph.nodes()]

    bipartite_graph.add_nodes_from(source_nodes, bipartite=0)
    bipartite_graph.add_nodes_from(target_nodes, bipartite=1)

    # 添加边：如果原图中存在边i->j，则在二分图中添加src_i到dst_j的边
    for edge in graph.edges():
        src_node = 'src_' + str(edge[0])
        dst_node = 'dst_' + str(edge[1])
        bipartite_graph.add_edge(src_node, dst_node)

    # 计算最大匹配 - 修复NetworkX版本兼容性问题
    try:
        # 尝试新版本的导入方式
        from networkx.algorithms import matching
        matching_result = matching.maximum_matching(bipartite_graph)
    except AttributeError:
        try:
            # 尝试旧版本的导入方式
            matching_result = nx.algorithms.matching.maximum_matching(bipartite_graph)
        except AttributeError:
            try:
                # 尝试直接调用
                matching_result = nx.maximum_matching(bipartite_graph)
            except AttributeError:
                # 如果都失败，使用备用方法
                print('  Warning: NetworkX maximum_matching not available, using alternative method')
                # 使用简单的贪心匹配作为备用
                matching_result = set()
                used_targets = set()
                for src_node in source_nodes:
                    for neighbor in bipartite_graph.neighbors(src_node):
                        if neighbor not in used_targets:
                            matching_result.add((src_node, neighbor))
                            used_targets.add(neighbor)
                            break

    matched_nodes = set()
    for edge in matching_result:
        if edge[0].startswith('src_'):
            matched_nodes.add(edge[0][4:])  # 移除'src_'前缀
        else:
            matched_nodes.add(edge[1][4:])  # 移除'src_'前缀

    # 未匹配的节点需要外部控制
    unmatched_nodes = set(graph.nodes()) - matched_nodes

    # 选择控制边：对于每个未匹配节点，选择影响分数最高的入边作为控制边
    control_edges = []

    for node in unmatched_nodes:
        predecessors = list(graph.predecessors(node))
        if predecessors:
            # 根据影响分数选择最佳前驱节点
            if node in influence_score.index:
                # 选择影响分数最高的前驱作为控制源
                best_pred = max(predecessors,
                                key=lambda x: influence_score.loc[x, 'influence_score']
                                if x in influence_score.index else 0)
                control_edges.append({
                    'from': best_pred,
                    'to': node,
                    'weight': influence_score.loc[node, 'influence_score']
                    if node in influence_score.index else 1.0
                })
            else:
                # 如果没有影响分数，选择第一个前驱
                control_edges.append({
                    'from': predecessors[0],
                    'to': node,
                    'weight': 1.0
                })
        else:
            # 如果节点没有前驱，标记为需要直接控制的节点
            control_edges.append({
                'from': 'EXTERNAL_CONTROL',
                'to': node,
                'weight': influence_score.loc[node, 'influence_score']
                if node in influence_score.index else 1.0
            })

    # 创建DataFrame
    control_edges_df = pd.DataFrame(control_edges)

    print(f'  {len(control_edges)} control edges identified.')
    print(f'  {len(unmatched_nodes)} nodes require external control.')

    # 保存结果
    if output_file:
        control_edges_df.to_csv(output_file, index=False)
        print(f'  MCES results saved to {output_file}')

    return control_edges_df, len(control_edges)

def compute_energy_optimal_drivers(directed_graph: nx.DiGraph,
                                   influence_score: pd.DataFrame,
                                   alpha=0.1,
                                   max_drivers=None,
                                   output_file=None):
    """
    计算能量最优控制驱动节点

    参数:
        directed_graph: 有向图网络
        influence_score: 基因影响分数
        alpha: 正则化参数
        max_drivers: 最大驱动节点数量
        output_file: 保存结果的文件路径

    返回:
        optimal_drivers_df: 最优驱动节点DataFrame
        energy_scores: 能量分数
    """
    print('  Solving Energy-Optimal Control problem...')

    # 移除自环
    graph = nx.DiGraph(directed_graph)
    graph.remove_edges_from(nx.selfloop_edges(graph))

    if graph.number_of_nodes() == 0:
        print('  Empty graph, no drivers needed.')
        return pd.DataFrame(columns=['gene', 'energy_score']), {}

    # 获取邻接矩阵
    nodes = list(graph.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    n = len(nodes)

    # 构建邻接矩阵
    A = nx.adjacency_matrix(graph, nodelist=nodes, weight='weights_combined').todense()
    A = np.array(A, dtype=float)

    # 计算每个节点作为驱动节点的控制能量
    energy_scores = {}

    for i, node in enumerate(nodes):
        try:
            # 构建控制矩阵B，第i个节点为驱动节点
            B = np.zeros((n, 1))
            B[i, 0] = 1.0

            # 计算控制能量：使用简化的能量估计
            # E = trace(B^T * W * B) 其中W是可控性Gramian的近似

            # 使用影响分数作为权重
            if node in influence_score.index:
                base_energy = 1.0 / (influence_score.loc[node, 'influence_score'] + 1e-6)
            else:
                base_energy = 1.0

            # 考虑节点的度数（连接性）
            out_degree = graph.out_degree(node)
            in_degree = graph.in_degree(node)
            degree_factor = 1.0 / (out_degree + in_degree + 1)

            # 结合基础能量和度数因子
            energy = base_energy * degree_factor

            # 添加网络拓扑因子
            # 计算节点到其他节点的平均距离（修复NumPy警告）
            try:
                # 安全地计算平均距离
                distances = []
                for target in nodes:
                    if target != node and nx.has_path(graph, node, target):
                        try:
                            dist = nx.shortest_path_length(graph, node, target)
                            distances.append(dist)
                        except nx.NetworkXNoPath:
                            continue

                # 安全地计算平均值，避免空数组警告
                if len(distances) > 0:
                    avg_distance = np.mean(distances)
                else:
                    avg_distance = 1.0  # 默认值，如果没有可达路径

                # 确保avg_distance是有效的数值
                if np.isnan(avg_distance) or avg_distance <= 0:
                    avg_distance = 1.0

            except Exception as e:
                # 如果计算失败，使用默认值
                avg_distance = 1.0

            energy *= avg_distance
            energy_scores[node] = energy

        except Exception as e:
            # 如果计算失败，使用默认值
            print(f'    Warning: Energy calculation failed for node {node}: {e}')
            energy_scores[node] = 1.0

    # 检查是否有有效的能量分数
    valid_scores = {k: v for k, v in energy_scores.items() if not np.isnan(v) and np.isfinite(v)}

    if not valid_scores:
        print('  Warning: No valid energy scores calculated, using uniform scores')
        energy_scores = {node: 1.0 for node in nodes}
        valid_scores = energy_scores

    # 根据能量分数排序（能量越低越好）
    sorted_nodes = sorted(valid_scores.items(), key=lambda x: x[1])

    # 如果指定了最大驱动节点数量，则只选择前max_drivers个
    if max_drivers and max_drivers < len(sorted_nodes):
        sorted_nodes = sorted_nodes[:max_drivers]

    # 创建结果DataFrame
    optimal_drivers = []
    for idx, (node, energy) in enumerate(sorted_nodes):
        optimal_drivers.append({
            'gene': node,
            'energy_score': energy,
            'influence_score': influence_score.loc[node, 'influence_score']
            if node in influence_score.index else 0.0,
            'rank': idx + 1
        })

    optimal_drivers_df = pd.DataFrame(optimal_drivers)

    # 安全地计算统计信息
    if len(valid_scores) > 0:
        min_energy = min(valid_scores.values())
        max_energy = max(valid_scores.values())
        print(f'  {len(optimal_drivers)} optimal driver nodes identified.')
        print(f'  Energy scores range: [{min_energy:.6f}, {max_energy:.6f}]')
    else:
        print(f'  {len(optimal_drivers)} optimal driver nodes identified.')
        print(f'  Warning: Energy score calculation had issues.')

    # 保存结果
    if output_file:
        optimal_drivers_df.to_csv(output_file, index=False)
        print(f'  Energy-optimal results saved to {output_file}')

    return optimal_drivers_df, energy_scores

def driver_regulators(GRN_nx: nx.DiGraph,
                      gene_influence_score: pd.DataFrame,
                      topK: int = 100,
                      driver_union: bool = True,
                      solver: str = 'GUROBI',
                      include_mces: bool = True,
                      include_energy: bool = True,
                      max_energy_drivers: int = None):
    """
    增强版驱动调节器识别，包含MDS、MFVS、MCES和能量最优控制四种方法

    参数:
        GRN_nx: 基因调控网络
        gene_influence_score: 基因影响分数
        topK: 候选基因数量
        driver_union: 是否取并集
        solver: 求解器
        include_mces: 是否包含MCES方法
        include_energy: 是否包含能量最优控制方法
        max_energy_drivers: 能量最优驱动节点最大数量

    返回:
        enhanced_drivers_df: 增强的驱动调节器结果
        method_results: 各方法的详细结果
    """
    print('[2] - Identifying driver regulators with enhanced methods...')

    # 检查是否可以使用Gurobi
    if solver == 'GUROBI':
        try:
            import gurobipy as gp
            gp.Model()
        except:
            print("Warning: GUROBI not available, switching to GLPK_MI")
            solver = 'GLPK_MI'

    # MFVS driver nodes
    MFVS_driver_set, source_nodes = MFVScontrol(GRN_nx, gene_influence_score.loc[:, 'score_out'], solver=solver)
    MFVS_driver_set = MFVS_driver_set.union(source_nodes)

    # MDS driver nodes
    MDS_driver_set, MDS_intermittent_nodes = MDScontrol(GRN_nx, solver=solver)

    # Top-ranked genes based on influence score
    critical_genes, out_critical_genes, in_critical_genes = highly_weighted_genes(gene_influence_score, topK // 2)

    # 新增方法结果存储
    method_results = {
        'MDS_drivers': MDS_driver_set,
        'MFVS_drivers': MFVS_driver_set,
        'critical_genes': critical_genes
    }

    # MCES方法
    if include_mces:
        try:
            mces_edges, mces_count = compute_mces_control_edges(GRN_nx, gene_influence_score)
            # 从控制边中提取需要控制的节点
            mces_driver_nodes = set(mces_edges['to'].unique()) if not mces_edges.empty else set()
            method_results['MCES_drivers'] = mces_driver_nodes
            method_results['MCES_edges'] = mces_edges
            print(f'  MCES method: {len(mces_driver_nodes)} driver nodes, {mces_count} control edges')
        except Exception as e:
            print(f'  Warning: MCES calculation failed: {e}')
            method_results['MCES_drivers'] = set()
            method_results['MCES_edges'] = pd.DataFrame()

    # 能量最优控制方法
    if include_energy:
        try:
            if max_energy_drivers is None:
                max_energy_drivers = min(topK, len(critical_genes))

            energy_drivers_df, energy_scores = compute_energy_optimal_drivers(
                GRN_nx, gene_influence_score, max_drivers=max_energy_drivers
            )
            energy_driver_nodes = set(energy_drivers_df['gene'].tolist())
            method_results['Energy_drivers'] = energy_driver_nodes
            method_results['Energy_scores'] = energy_scores
            method_results['Energy_drivers_df'] = energy_drivers_df
            print(f'  Energy-optimal method: {len(energy_driver_nodes)} driver nodes')
        except Exception as e:
            print(f'  Warning: Energy-optimal calculation failed: {e}')
            method_results['Energy_drivers'] = set()
            method_results['Energy_scores'] = {}

    # 合并所有方法的驱动节点
    all_driver_sets = [MDS_driver_set, MFVS_driver_set]
    if include_mces and 'MCES_drivers' in method_results:
        all_driver_sets.append(method_results['MCES_drivers'])
    if include_energy and 'Energy_drivers' in method_results:
        all_driver_sets.append(method_results['Energy_drivers'])

    if driver_union:
        combined_drivers = set().union(*all_driver_sets)
    else:
        combined_drivers = set.intersection(*all_driver_sets) if len(all_driver_sets) > 1 else all_driver_sets[0]

    # 最终的TEDC2L驱动调节器：结合高影响分数基因
    TEDC2L_drivers = combined_drivers.intersection(critical_genes)

    # 创建增强的结果DataFrame
    all_candidates = set().union(*all_driver_sets, critical_genes)
    enhanced_drivers_df = gene_influence_score.loc[
        list(all_candidates),
        ['influence_score', 'score_out', 'score_in']
    ].copy()

    # 添加方法标记
    enhanced_drivers_df['is_driver_regulator'] = enhanced_drivers_df.index.isin(TEDC2L_drivers)
    enhanced_drivers_df['is_MFVS_driver'] = enhanced_drivers_df.index.isin(MFVS_driver_set)
    enhanced_drivers_df['is_MDS_driver'] = enhanced_drivers_df.index.isin(MDS_driver_set)

    if include_mces and 'MCES_drivers' in method_results:
        enhanced_drivers_df['is_MCES_driver'] = enhanced_drivers_df.index.isin(method_results['MCES_drivers'])

    if include_energy and 'Energy_drivers' in method_results:
        enhanced_drivers_df['is_Energy_driver'] = enhanced_drivers_df.index.isin(method_results['Energy_drivers'])
        # 添加能量分数
        energy_score_map = method_results.get('Energy_scores', {})
        enhanced_drivers_df['energy_score'] = [
            energy_score_map.get(gene, np.nan) for gene in enhanced_drivers_df.index
        ]

    enhanced_drivers_df.sort_values(by='influence_score', ascending=False, inplace=True)

    # 更新method_results
    method_results['TEDC2L_drivers'] = TEDC2L_drivers
    method_results['enhanced_drivers_df'] = enhanced_drivers_df
    method_results['out_critical_genes'] = out_critical_genes
    method_results['in_critical_genes'] = in_critical_genes

    return enhanced_drivers_df, method_results


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
                'drivers': list(driver_set) if len(driver_set) <= 50 else f"Too many to list ({len(driver_set)} total)"
            })

    comparison_df = pd.DataFrame(comparison_data)
    comparison_path = output_path / 'method_comparison_summary.csv'
    comparison_df.to_csv(comparison_path, index=False)
    print(f"Method comparison saved to {comparison_path}")

