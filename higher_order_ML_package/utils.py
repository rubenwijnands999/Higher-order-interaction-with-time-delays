import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import networkx as nx
from collections import defaultdict
from itertools import combinations, permutations, combinations_with_replacement


def cost_fn_efficient(G, H, y, reg_lambda = None):
    return 1/(len(y)) * np.linalg.norm(y - np.ones(G.shape[0]).T @ G)**2 + reg_lambda * np.sum(H,axis=(0,1))

def compute_metric(interactions,ground_truth,metric_name='MSE'):
    """
    Compute the metric for the estimated interactions
    """
    if metric_name == 'MSE':
        return compute_MSE_metric(interactions,ground_truth)
    elif metric_name == 'ROC_AUC':
        return compute_ROC_AUC_metric(interactions,ground_truth)
    else:
        raise ValueError(f"Metric {metric_name} not recognized")

def compute_MSE_metric(interactions,ground_truth):
    """
    Compute the MSE metric for the estimated interactions
    """
    MSE = {}
    for target_node in interactions.keys():
        MSE[target_node] = 0
        assert len(interactions[target_node]) == len(ground_truth[target_node]), "Number of interactions do not match"
        num_interactions = len(interactions[target_node])
        for i in range(num_interactions):
            MSE[target_node] += (interactions[target_node][i][1] - ground_truth[target_node][i][1])**2
        MSE[target_node] = MSE[target_node]/num_interactions
    return MSE

def compute_ROC_AUC_metric(interactions,ground_truth):
    """
    Compute the ROC AUC metric for the estimated interactions
    """
    ROC_AUC = {}
    for target_node in interactions.keys():
        num_interactions = len(interactions[target_node])-1  # Exclude bias term
        assert len(interactions[target_node]) == len(ground_truth[target_node]), "Number of interactions do not match"
        all_labels = []
        all_weights = []
        for i in range(num_interactions):
            all_labels.append(ground_truth[target_node][i][1])
            all_weights.append(interactions[target_node][i][1])
        ROC_AUC[target_node] = roc_auc_score(all_labels, all_weights)
    return ROC_AUC


# def gen_ground_truth_interactions(HOI,D,N,weight=1):
#     """
#     Generate ground truth from interactions
#     """
#     all_interactions_gt = []
#     for interaction_size in range(1, D + 1):
#         for interaction in combinations_with_replacement([i for i in range(N) if i != 0], interaction_size):
#             if list(interaction) in HOI[0]:
#                 all_interactions_gt.append( (list(interaction),weight) )
#             else:
#                 all_interactions_gt.append( (list(interaction),0) )
#     # add bias term
#     all_interactions_gt.append((['bias'],0))
#     return all_interactions_gt

def gen_ground_truth_interactions(HOI,D,N,weight_input=1,C=None):
    """
    Generate ground truth from interactions when we have matrix C as well
    """
    if C is None:
        C = np.ones((N,1))
    M = int(C.shape[1] - 1)
    all_interactions_gt = []
    for interaction_size in range(1, D + 1):
        for interaction in combinations_with_replacement([i for i in range(1, (N-1)*(M+1)+1)], interaction_size):
            reduced_interaction = [i % (N-1) + 1 if i>= N else i for i in interaction]
            if 0 not in reduced_interaction:
                C_index = [i // N for i in interaction]
                prod = np.prod(C[reduced_interaction,C_index], axis=0)
                weight = weight_input if list(interaction) in HOI[0] else 0
                all_interactions_gt.append( (list(interaction),weight * prod) )
    # add bias term
    all_interactions_gt.append((['bias'],0))
    return all_interactions_gt

def reorder_all_weights(weights,C,ALS=False):
    """
    Reorder weights to match the ground truth order of the interactions.
    :param weights:
    :param M:
    :return:
    """
    N,M = C.shape
    new_weights = []
    for i in range(len(weights)):
        res = []
        # if not ALS:
        #     res.append(M*weights[i][0,:][None,:])
        #     for j in range(M):
        #         res.append(weights[i][M+j::M,:])
        # else:
        res.append(weights[i][0,:][None,:])
        for j in range(M):
            res.append(weights[i][1+j::M,:])
        new_weights.append(np.concatenate(res, axis=0))
    return new_weights


def find_index_permutations(input_list):
    # Generate all permutations and convert to NumPy arrays
    permuted_arrays = [np.array(p) for p in permutations(input_list)]

    # Include the original index array
    result = [input_list] + permuted_arrays

    # Remove duplicates (optional, if index may already match a permutation)
    input_list = [np.array(x) for x in set(tuple(row) for row in result)]

    return input_list

def construct_index(interaction, D):
    index = np.zeros(D, dtype=int) # Has to be of length D

    if type(interaction) == list:
        interaction = np.array(interaction)

    interaction = 1 + np.where(interaction > 0, interaction-1, interaction)
    index[:len(interaction)] = interaction

    index = find_index_permutations(index)

    return index


def query_weight(W,index):
    if type(index) == list:
        res = 0
        for idx in index:
            res += query_weight_computation(W, idx)
    else:
        res = query_weight_computation(W, index)
    return res

def query_weight_computation(W,index):
    """
    Query the weight matrix for a specific index.
    """
    assert len(W) == len(index), "The number of indices must match the number of dimensions"
    rank = np.array(W[0]).shape[1]
    res = 0
    for r in range(rank):
        prod_res = 1
        for d in range(len(W)):
            prod_res *= W[d][index[d],r]
        res += prod_res
    return res


def get_all_unique_possible_interactions(N, D):
    interaction_list = []
    for interaction_size in range(1, D + 1):
        for interaction in combinations_with_replacement([i for i in range(N) if i != 0], interaction_size):
            interaction_list.append( list(interaction) )
    return interaction_list

def get_unique_interactions(W,D):
    interactions_Volterra_CP = []
    N = W[0].shape[0]
    all_interactions = get_all_unique_possible_interactions(N, D)
    for i,interaction in enumerate(all_interactions):
        if D == 1:
            interactions_Volterra_CP.append((interaction,W[0][i+1]))
        else:
            index = construct_index(interaction, D)
            weight = query_weight(W, index)
            interactions_Volterra_CP.append((interaction, weight))
    # Add bias term
    if D != 1:
        bias = query_weight(W, np.zeros(D).astype(int) )
        interactions_Volterra_CP.append((['bias'], bias))
    return interactions_Volterra_CP


def create_optimal_node_layout(all_interactions, node_map, N,weight_threshold=0.1, layout_quality=1.0, cluster_strength=5.0,scale=1):
    """Compute positions with clustered higher-order interactions."""
    H = nx.Graph()

    # Add real nodes
    for node in range(N):
        H.add_node(node, label=node_map.get(node))

    virtual_nodes = {}
    interaction_groups = defaultdict(list)

    # Add virtual nodes with enhanced clustering
    for target_node, interactions in all_interactions.items():
        for interaction, weight in interactions:
            if abs(weight) >= weight_threshold:
                interaction_tuple = frozenset(interaction)
                key = (target_node, interaction_tuple)

                if key not in virtual_nodes:
                    virtual_node_id = len(H)
                    H.add_node(virtual_node_id, virtual=True)
                    virtual_nodes[key] = virtual_node_id

                    # Stronger connections to interaction members
                    for node in interaction:
                        H.add_edge(node, virtual_node_id,
                                   weight=1)#weight * cluster_strength)

                    # Connect interaction members to each other
                    for u, v in combinations(interaction, 2):
                        H.add_edge(u, v, weight=1)#weight * 0.7)

                # Connect target to virtual node
                H.add_edge(target_node, virtual_nodes[key],
                           weight=1)#weight * cluster_strength)

                interaction_groups[interaction_tuple].append(target_node)

    # Extra clustering force for nodes in multiple interactions
    for interaction, targets in interaction_groups.items():
        if len(targets) > 1:
            for u, v in combinations(targets, 2):
                # if H.has_edge(u, v):
                #     H[u][v]['weight'] *= 1.5  # Stronger boost for existing
                # else:
                H.add_edge(u, v, weight=1.0 )#* cluster_strength)

    # Calculate adaptive layout parameters
    n_nodes = H.number_of_nodes()
    base_k = 1.0 / np.sqrt(n_nodes) if n_nodes > 0 else 1.0
    k = base_k * layout_quality * 2  # Adjusted multiplier

    # Add weak background connections to prevent circular component layouts
    if nx.number_connected_components(H) > 1:
        H.add_edges_from(combinations(H.nodes(), 2), weight=0.001)

    return nx.spring_layout(
        H,
        weight="weight",
        k=k,
        iterations=500,  # Increased iterations for convergence
        scale=scale,      # Larger scale for bigger graphs
        seed=42
    )


def plot_graph(all_interactions, N,weight_threshold=0.1, layout_quality=1.0, cluster_strength=5.0, scale=1):
    """
    Parameters:
    - weight_threshold: Minimum absolute weight to display
    - layout_quality: Multiplier for layout spacing (higher = more spread out)
    """

    node_map = {i: i for i in range(N)}

    G = nx.Graph()
    for node in range(N):
        G.add_node(node, label=node_map.get(node, str(node)))

    pos = create_optimal_node_layout(all_interactions, node_map, N,
                                     weight_threshold=weight_threshold, layout_quality=layout_quality,
                                     cluster_strength=cluster_strength, scale=1)

    plt.figure(figsize=(14, 10))
    ax = plt.gca()

    # Node styling
    nx.draw_networkx_nodes(
        G, pos,
        node_size=1200,
        node_color='lightblue',
        edgecolors='black',
        linewidths=1.5,
        alpha=0.9
    )
    nx.draw_networkx_labels(
        G, pos,
        labels={n: node_map.get(n, str(n)) for n in G.nodes},
        font_size=12,
        font_weight='bold'
    )

    # Collect weights for normalization
    all_weights = [
                      w for target, interactions in all_interactions.items()
                      for interaction, w in interactions
                      if abs(w) >= weight_threshold
                  ]

    # Symmetric color normalization
    if all_weights:
        max_abs = max(abs(w) for w in all_weights)
        cmap = plt.cm.coolwarm
        norm = plt.Normalize(vmin=-max_abs, vmax=max_abs)
    else:
        cmap, norm = None, None

    # Draw higher-order interactions
    for target_node, interactions in all_interactions.items():
        for interaction, weight in interactions:
            if abs(weight) < weight_threshold:
                continue

            if len(interaction) == 1:
                color = cmap(norm(weight)) if cmap else 'gray'
                width = norm(abs(weight)) * 6 + 1
                ax.plot(
                    [pos[interaction[0]][0], pos[target_node][0]],
                    [pos[interaction[0]][1], pos[target_node][1]],
                    color=color,
                    linewidth=width,
                    linestyle='-',
                    alpha=0.9,
                    solid_capstyle='round'
                )
            else:
                color = cmap(norm(weight)) if cmap else 'gray'
                width = norm(abs(weight)) * 6 + 1

                # Calculate interaction hub position
                points = np.array([pos[n] for n in interaction])
                centroid = points.mean(axis=0)

                # Draw lines from interaction nodes to hub
                for node in interaction:
                    ax.plot(
                        [pos[node][0], centroid[0]],
                        [pos[node][1], centroid[1]],
                        color=color,
                        linewidth=width * 0.7,
                        linestyle=':',
                        alpha=0.6,
                        solid_capstyle='round'
                    )

                # Draw line from target node to hub
                ax.plot(
                    [pos[target_node][0], centroid[0]],
                    [pos[target_node][1], centroid[1]],
                    color=color,
                    linewidth=width,
                    linestyle='-',
                    alpha=0.9,
                    solid_capstyle='round'
                )

    # Symmetric colorbar
    if all_weights:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
        cbar.set_label('Interaction Strength', fontsize=12)

    # Legend
    legend_elements = [
        mlines.Line2D([], [], color='black', linewidth=3, label='Direct Connection'),
        mlines.Line2D([], [], color='black', linewidth=3, linestyle=':',
                      label='Interaction Members'),
        mlines.Line2D([], [], color='black', linewidth=3, linestyle='-',
                      label='To Interaction Hub')
    ]
    #ax.legend(handles=legend_elements, loc='best', fontsize=10)

    plt.title("Network Interactions with Threshold Filtering", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
