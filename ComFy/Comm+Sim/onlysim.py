import networkx as nx
import nx_cugraph as nxcg
import numpy as np
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.utils import to_networkx, from_networkx
import random
from sklearn.metrics.cluster import normalized_mutual_info_score as NMI



def modify_graph(data, dataset_name, budget_edges_add, budget_edges_delete, seed):

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

    # Calculate similarity between all pairs of nodes
    features = data.x.numpy()
    similarities = cosine_similarity(features)
    sim = np.mean([similarities[u, v] for u, v in G.edges()])

    ranking = {}
    ranking_add = {}
    ranking_remove = {}
    for u, v in G.edges(): # delete edges
        if u == v: continue
        if sim < similarities[u, v]: continue # delete only edges with similarity lower than average
        ranking[(u, v)] = (sim*original_edge_count - similarities[u, v])/(original_edge_count - 1)
        ranking_remove[(u, v)] = ranking[(u, v)]
    for u, v in nx.non_edges(G):
        if u == v: continue
        if sim > similarities[u, v]: continue # add only edges with similarity higher than average
        ranking[(u, v)] = (sim*original_edge_count + similarities[u, v])/(original_edge_count + 1)
        ranking_add[(u, v)] = ranking[(u, v)]

    ranking_add_s = {k: v for k, v in sorted(ranking_add.items(), key=lambda item: item[1], reverse=True)}
    ranking_remove_s = {k: v for k, v in sorted(ranking_remove.items(), key=lambda item: item[1], reverse=True)}

    edges_added = []
    edges_removed = []
    for (u, v), sim in ranking_add_s.items():
        if len(edges_added) >= budget_edges_add:
            break
        if G.has_edge(u, v): continue
        G.add_edge(u, v)
        edges_added.append((u, v))
    for (u, v), sim in ranking_remove_s.items():
        if len(edges_removed) >= budget_edges_delete:
            break
        if G.has_edge(u, v):
            G.remove_edge(u, v)
            edges_removed.append((u, v))

    # logging information about rewired edges
    # count additions or deletions
    added_edges = len(edges_added)
    deleted_edges = len(edges_removed)

    # create len(communities) x len(communities) matrix to store number of edges between each pair of clusters
    num_edges_rewired_per_comm = {(i, j): 0 for i in range(len(communities)) for j in range(i, len(communities))}
    for u, v in [edge for edge in edges_added + edges_removed]:
        comm_i = [i for i, comm in enumerate(communities) if u in comm][0]
        comm_j = [i for i, comm in enumerate(communities) if v in comm][0]
        if comm_i < comm_j:
            num_edges_rewired_per_comm[(comm_i, comm_j)] += 1
        else:
            num_edges_rewired_per_comm[(comm_j, comm_i)] += 1
    
    # new matrix but normalized by the number of edges in the original community
    num_edges_rewired_per_comm_normalized = {(i, j): 
        num_edges_rewired_per_comm[(i, j)] / (len(communities[i]) * len(communities[j])) for i, j in num_edges_rewired_per_comm}
    
    # LOGGING
    mean = np.mean(list(num_edges_rewired_per_comm_normalized.values()))
    cov = np.cov(list(num_edges_rewired_per_comm_normalized.values()))
    # now separated by diagonal and off-diagonal
    mean_diag = np.mean([num_edges_rewired_per_comm_normalized[(i, i)] for i in range(len(communities))])
    mean_off_diag = np.mean([num_edges_rewired_per_comm_normalized[(i, j)] for i in range(len(communities)) for j in range(i+1, len(communities))])
    cov_diag = np.cov([num_edges_rewired_per_comm_normalized[(i, i)] for i in range(len(communities))])
    cov_off_diag = np.cov([num_edges_rewired_per_comm_normalized[(i, j)] for i in range(len(communities)) for j in range(i+1, len(communities))])


    sim2 = np.mean([similarities[u, v] for u, v in G.edges()])
    print("similarity before and after",sim,sim2)
    pyg_data = from_networkx(G)
    newG = to_networkx(pyg_data, to_undirected=True)
    nxcg_G = nxcg.from_networkx(newG) 
    communities_after = list(nx.community.louvain_communities(nxcg_G, seed=seed))
    cluster_dict_after = {node: i for i, cluster in enumerate(communities_after) for node in cluster}
    cluster_list_after = [cluster_dict_after[node] for node in range(len(data.y))]
    nmiscoremod_after = NMI(cluster_list_after, data.y.cpu().numpy())

    print(f"Number of edges rewired: {added_edges + deleted_edges}")
    print(f"Mean of rewired edges by cluster: {mean}")
    print(f"Covariance of rewired edges by cluster: {cov}")
    print(f"Mean of rewired edges intra-cluster: {mean_diag}")
    print(f"Covariance of rewired edges intra-cluster: {cov_diag}")
    print(f"Mean of rewired edges inter-cluster: {mean_off_diag}")
    print(f"Covariance of rewired edges inter-cluster: {cov_off_diag}")
    print(f"NMI before: {nmiscoremod_before}")
    print(f"NMI after: {nmiscoremod_after}")




    # Prepare data for CSV
    csv_data = {
        'Metric': [
            'Dataset',
            'Number of Communities',
            # 'Average Intra-Cluster Similarity',
            'Edges to be Added',
            'Edges to be Removed',
            # 'Original Edge Count',
            # 'Expected New Edge Count'
            'Mean edges by cluster',
            'Covariance edges by cluster',
            'Mean edges intra-cluster',
            'Covariance edges intra-cluster',
            'Mean edges inter-cluster',
            'Covariance edges inter-cluster',
            'NMI before',
            'NMI after'
        ],
        'Value': [
            dataset_name,
            len(communities),
            added_edges,
            deleted_edges,
            mean,
            cov,
            mean_diag,
            cov_diag,
            mean_off_diag,
            cov_off_diag,
            nmiscoremod_before,
            nmiscoremod_after

        ]
    }


    # Create DataFrame and save to CSV
    df = pd.DataFrame(csv_data)
    csv_filename = f"simrewirings/similarityreports/{dataset_name}_{budget_edges_add}_{budget_edges_delete}_onlysim.csv"
    df.to_csv(csv_filename, index=False)
    print(f"CSV file '{csv_filename}' has been created.")



    # Ensure we keep the node features, labels, and masks from the original data
    pyg_data.x = data.x
    pyg_data.y = data.y
    pyg_data.train_mask = data.train_mask
    pyg_data.val_mask = data.val_mask
    pyg_data.test_mask = data.test_mask

    print(pyg_data)

    return pyg_data,nmiscoremod_before,nmiscoremod_after