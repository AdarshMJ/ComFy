import torch
import networkx as nx
import nx_cugraph as nxcg
import numpy as np
import pandas as pd
import os
from torch_geometric.utils import to_networkx, from_networkx
from sklearn.metrics import normalized_mutual_info_score as NMI
from tqdm import tqdm
import itertools

def modify_graph(data, dataset_name, budget_add, budget_delete, seed):

    # Extract dataset name from data object
    dataset_name, _ = os.path.splitext(dataset_name)
    print("=============================================================")
    print("Rewiring based on feature similarity...")
    G = to_networkx(data, to_undirected=True)
    nxcg_G = nxcg.from_networkx(G) 
    communities = list(nx.community.louvain_communities(nxcg_G, seed=seed))
    cluster_dict_before = {node: i for i, cluster in enumerate(communities) for node in cluster}
    cluster_list_before = [cluster_dict_before[node] for node in range(len(data.y))]
    nmiscoremod_before = NMI(cluster_list_before, data.y.cpu().numpy())     
    original_edge_count = G.number_of_edges()

    # Move data to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    features = data.x.to(device)

    # Calculate similarity between all pairs of nodes using GPU
    similarities = torch.mm(features, features.t())
    norms = torch.norm(features, dim=1, keepdim=True)
    similarities = similarities / (norms * norms.t())
    similarities = similarities.cpu().numpy()  # Move back to CPU for NetworkX operations

    if np.isnan(similarities).any():
        print("NaN detected in similarities matrix")
    # budgets per community (normalized)
    scores = { (i, j): (len(comm1) * len(comm2)) 
            for i, comm1 in enumerate(communities) for j, comm2 in enumerate(communities) if i <= j }
    norm_scores = { (i, j): scores[(i, j)] / sum(scores.values()) for i, j in scores }
    budgets_add = { (i, j): int(budget_add * norm_scores[(i, j)] ) for i, j in norm_scores }
    budgets_delete = { (i, j): int(budget_delete * norm_scores[(i, j)] ) for i, j in norm_scores }

    edges_added = set()
    edges_removed = set()

    total_comparisons = sum(1 for i in range(len(communities)) for j in range(i, len(communities)))
    with tqdm(total=total_comparisons, desc="Processing communities") as pbar:
        for i, comm1 in enumerate(communities):
            for j, comm2 in enumerate(communities[i:], start=i):
                if i > j:
                    continue  # Skip redundant comparisons

                # Convert communities to sets for faster membership checking
                comm1_set = set(comm1)
                comm2_set = set(comm2)

                # Use set operations to find edges between communities
                edges_between = set(G.edges(comm1_set)) & set((u, v) for u in comm1_set for v in comm2_set)
                if not edges_between:
                    pbar.update(1)
                    continue  # Skip if no edges between comm1 and comm2

                # Use NumPy for faster array operations
                comm1_arr = np.array(list(comm1_set))
                comm2_arr = np.array(list(comm2_set))
                sim_matrix = similarities[comm1_arr[:, None], comm2_arr]

                # Calculate sim using vectorized operations
                edge_indices = np.array([(comm1_arr.tolist().index(u), comm2_arr.tolist().index(v)) for u, v in edges_between])
                sim_values = sim_matrix[edge_indices[:, 0], edge_indices[:, 1]]
                sim = np.mean(sim_values) if sim_values.size > 0 else 0

                num_edges = len(edges_between)

                # Compute rankings using vectorized operations
                ranking_remove = [(u, v, (sim*num_edges - similarities[u, v])/(num_edges - 1)) 
                                  for u, v in edges_between if similarities[u, v] < sim]
                
                # Use set difference for non_edges
                non_edges = set(itertools.product(comm1_set, comm2_set)) - set(G.edges()) - {(v, u) for u, v in G.edges()}
                ranking_add = [(u, v, (sim*num_edges + similarities[u, v])/(num_edges + 1)) 
                               for u, v in non_edges if similarities[u, v] > sim]

                # Use NumPy for sorting
                ranking_add_s = np.array(ranking_add, dtype=[('u', int), ('v', int), ('score', float)])
                ranking_remove_s = np.array(ranking_remove, dtype=[('u', int), ('v', int), ('score', float)])
                ranking_add_s.sort(order='score')
                ranking_remove_s.sort(order='score')

                # Add edges
                edges_to_add = min(budgets_add[(i, j)], len(ranking_add_s))
                for u, v, _ in ranking_add_s[-edges_to_add:]:
                    if (u, v) not in edges_added and (v, u) not in edges_added:
                        if len(edges_added) < budget_add:
                            G.add_edge(u, v)
                            edges_added.add((min(u, v), max(u, v)))  # Store only one direction
                        else:
                            break

                # Remove edges
                edges_to_remove = min(budgets_delete[(i, j)], len(ranking_remove_s))
                for u, v, _ in ranking_remove_s[-edges_to_remove:]:
                    if (u, v) in G.edges() or (v, u) in G.edges():
                        if len(edges_removed) < budget_delete:
                            G.remove_edge(u, v)
                            edges_removed.add((min(u, v), max(u, v)))  # Store only one direction
                        else:
                            break

                pbar.update(1)

    final_edge_count = G.number_of_edges()
    edges_modified = final_edge_count - original_edge_count

    # logging information about rewired edges
    added_edges = len(edges_added)
    deleted_edges = len(edges_removed)
    total_edges = G.number_of_edges()

    # create len(communities) x len(communities) matrix to store number of edges between each pair of clusters
    num_edges_rewired_per_comm = {(i, j): 0 for i in range(len(communities)) for j in range(i, len(communities))}
    for u, v in edges_added.union(edges_removed):
        comm_i = [i for i, comm in enumerate(communities) if u in comm][0]
        comm_j = [i for i, comm in enumerate(communities) if v in comm][0]
        if comm_i < comm_j:
            num_edges_rewired_per_comm[(comm_i, comm_j)] += 1
        else:
            num_edges_rewired_per_comm[(comm_j, comm_i)] += 1
    
    # new matrix but normalized by the number of edges in the original community
    num_edges_rewired_per_comm_normalized = {(i, j): 
        num_edges_rewired_per_comm[(i, j)] / (len(communities[i]) * len(communities[j])) for i, j in num_edges_rewired_per_comm}
    if np.isnan(list(num_edges_rewired_per_comm_normalized.values())).any():
        print("NaN detected in num_edges_rewired_per_comm_normalized")

    # LOGGING
    mean = np.mean(list(num_edges_rewired_per_comm_normalized.values()))
    cov = np.cov(list(num_edges_rewired_per_comm_normalized.values()))
    mean_diag = np.mean([num_edges_rewired_per_comm_normalized[(i, i)] for i in range(len(communities))])
    mean_off_diag = np.mean([num_edges_rewired_per_comm_normalized[(i, j)] for i in range(len(communities)) for j in range(i+1, len(communities))])
    cov_diag = np.cov([num_edges_rewired_per_comm_normalized[(i, i)] for i in range(len(communities))])
    cov_off_diag = np.cov([num_edges_rewired_per_comm_normalized[(i, j)] for i in range(len(communities)) for j in range(i+1, len(communities))])
    if np.isnan([mean, cov, mean_diag, mean_off_diag, cov_diag, cov_off_diag]).any():
        print("NaN detected in logging statistics")

    pyg_data = from_networkx(G)

    csv_data = {
        'Metric': [
            'Dataset',
            'Number of Communities',
            'Original Edge Count',
            'Final Edge Count',
            'Edges Modified',
            'Edges to be Added (Budget)',
            'Edges to be Removed (Budget)',
            'Mean edges by cluster',
            'Covariance edges by cluster',
            'Mean edges intra-cluster',
            'Covariance edges intra-cluster',
            'Mean edges inter-cluster',
            'Covariance edges inter-cluster'
        ],
        'Value': [
            dataset_name,
            len(communities),
            original_edge_count,
            final_edge_count,
            edges_modified,
            budget_add,
            budget_delete,
            mean,
            cov,
            mean_diag,
            cov_diag,
            mean_off_diag,
            cov_off_diag
        ]
    }

    # Create DataFrame and save to CSV
    df = pd.DataFrame(csv_data)
    csv_filename = f"{dataset_name}_{budget_add}_{budget_delete}_comsim.csv"
    df.to_csv(csv_filename, index=False)
    print(f"CSV file '{csv_filename}' has been created.")

    # Print summary
    print(f"Original edge count: {original_edge_count}")
    print(f"Final edge count: {final_edge_count}")
    print(f"Edges modified: {edges_modified}")
    print(f"Mean of rewired edges by cluster: {mean}")
    print(f"Covariance of rewired edges by cluster: {cov}")
    print(f"Mean of rewired edges intra-cluster: {mean_diag}")
    print(f"Covariance of rewired edges intra-cluster: {cov_diag}")
    print(f"Mean of rewired edges inter-cluster: {mean_off_diag}")
    print(f"Covariance of rewired edges inter-cluster: {cov_off_diag}")

    # Ensure we keep the node features, labels, and masks from the original data
    pyg_data.x = data.x
    pyg_data.y = data.y
    pyg_data.train_mask = data.train_mask
    pyg_data.val_mask = data.val_mask
    pyg_data.test_mask = data.test_mask

    print(pyg_data)

    return pyg_data, nmiscoremod_before