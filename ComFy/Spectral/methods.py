import time
import torch
import networkx as nx
from dataloader import *
from tqdm import tqdm
from rewiring.fastrewiringKupdates import *
#from rewiring.fastrewiringmax import *
from rewiring.MinGapKupdates import *
from rewiring.fosr import *
from rewiring.spectral_utils import *
from rewiring.sdrf import *
from torch_geometric.utils import to_networkx,from_networkx,homophily
import random
from clustering import *
from sklearn.metrics.cluster import normalized_mutual_info_score as NMI


def proxydelmin(data, nxgraph,seed, max_iterations):
    print("Deleting edges to minimize the gap...")
    
    
    # Track the original edges
    original_edges = set(nxgraph.edges())
    
    # Perform community detection before rewiring
    clustermod_before = maximize_modularity(nxgraph)
    cluster_dict_before = {node: i for i, cluster in enumerate(clustermod_before) for node in cluster}
    
    # Initialize counters before rewiring
    same_class_same_community_before = 0
    same_class_diff_community_before = 0
    diff_class_same_community_before = 0
    diff_class_diff_community_before = 0
    
    # Assuming `data.y` contains the node labels
    labels = data.y.cpu().numpy()
    
    start_algo = time.time()
    newgraph = min_and_update_edges(nxgraph, rank_by_proxy_delete_min, "proxydeletemin",seed, max_iter=max_iterations, updating_period=1)
    newgraph.remove_edges_from(list(nx.selfloop_edges(newgraph)))
    end_algo = time.time()
    # Track the edges after deletion
    updated_edges = set(newgraph.edges())
    
    # Determine the deleted edges
    deleted_edges = original_edges - updated_edges
    
    # Initialize counters after rewiring
    same_class_edges = 0
    diff_class_edges = 0
    same_class_same_community_after = 0
    same_class_diff_community_after = 0
    diff_class_same_community_after = 0
    diff_class_diff_community_after = 0
    
    # Perform community detection after rewiring
    clustermod_after = maximize_modularity(newgraph)
    cluster_dict_after = {node: i for i, cluster in enumerate(clustermod_after) for node in cluster}
    
    # Count same-class and different-class edges after rewiring
    for edge in deleted_edges:
        node1, node2 = edge
        same_class = labels[node1] == labels[node2]
        same_community_before = cluster_dict_before[node1] == cluster_dict_before[node2]
        same_community_after = cluster_dict_after[node1] == cluster_dict_after[node2]
        
        if same_class:
            same_class_edges += 1
            if same_community_after:
                same_class_same_community_after += 1 
            else:
                same_class_diff_community_after += 1
            if same_community_before:
                same_class_same_community_before += 1
            else:
                same_class_diff_community_before += 1
        else:
            diff_class_edges += 1
            if same_community_after:
                diff_class_same_community_after += 1
            else:
                diff_class_diff_community_after += 1
            if same_community_before:
                diff_class_same_community_before += 1
            else:
                diff_class_diff_community_before += 1
    
    fgap, _, _, _ = spectral_gap(newgraph)
    print()
    print(f"FinalGap = {fgap}") 
    data_modifying = end_algo - start_algo
    newdata = from_networkx(newgraph)
    print(newdata)
    #data.edge_index = torch.cat([newdata.edge_index])
    
    cluster_list = [cluster_dict_after[node] for node in range(len(data.y))]
    nmiscoremod = NMI(cluster_list, labels)
    
    return data, fgap, data_modifying, same_class_edges, diff_class_edges, nmiscoremod, same_class_same_community_after, same_class_diff_community_after, diff_class_same_community_after, diff_class_diff_community_after, same_class_same_community_before, same_class_diff_community_before, diff_class_same_community_before, diff_class_diff_community_before


def proxydelmax(data, nxgraph,seed, max_iterations):
    print("Deleting edges to maximize the gap...")
    start_algo = time.time()
    
    # Track the original edges
    original_edges = set(nxgraph.edges())
    
    # Perform community detection before rewiring
    clustermod_before = maximize_modularity(nxgraph)
    cluster_dict_before = {node: i for i, cluster in enumerate(clustermod_before) for node in cluster}
    
    # Initialize counters before rewiring
    same_class_same_community_before = 0
    same_class_diff_community_before = 0
    diff_class_same_community_before = 0
    diff_class_diff_community_before = 0
    
    # Assuming `data.y` contains the node labels
    labels = data.y.cpu().numpy()
    
    # Count same-class and different-class edges before rewiring
    # for edge in original_edges:
    #     node1, node2 = edge
    #     same_class = labels[node1] == labels[node2]
    #     same_community = cluster_dict_before[node1] == cluster_dict_before[node2]
        
    #     if same_class:
    #         if same_community:
    #             same_class_same_community_before += 1
    #         else:
    #             same_class_diff_community_before += 1
    #     else:
    #         if same_community:
    #             diff_class_same_community_before += 1
    #         else:
    #             diff_class_diff_community_before += 1
    start_algo = time.time()
    newgraph = process_and_update_edges(nxgraph, rank_by_proxy_delete, "proxydeletemax",seed, max_iter=max_iterations, updating_period=1)
    newgraph.remove_edges_from(list(nx.selfloop_edges(newgraph)))
    end_algo = time.time()
    # Track the edges after deletion
    updated_edges = set(newgraph.edges())
    
    # Determine the deleted edges
    deleted_edges = original_edges - updated_edges
    
    # Initialize counters after rewiring
    same_class_edges = 0
    diff_class_edges = 0
    same_class_same_community_after = 0
    same_class_diff_community_after = 0
    diff_class_same_community_after = 0
    diff_class_diff_community_after = 0
    
    # Perform community detection after rewiring
    clustermod_after = maximize_modularity(newgraph)
    cluster_dict_after = {node: i for i, cluster in enumerate(clustermod_after) for node in cluster}
    
    # Count same-class and different-class edges after rewiring
    for edge in deleted_edges:
        node1, node2 = edge
        same_class = labels[node1] == labels[node2]
    
        same_community_before = cluster_dict_before[node1] == cluster_dict_before[node2]
        same_community_after = cluster_dict_after[node1] == cluster_dict_after[node2]
        if same_class:
            same_class_edges += 1
            if same_community_after:
                same_class_same_community_after += 1 
            else:
                same_class_diff_community_after += 1
            if same_community_before:
                same_class_same_community_before += 1
            else:
                same_class_diff_community_before += 1
        else:
            diff_class_edges += 1
            if same_community_after:
                diff_class_same_community_after += 1
            else:
                diff_class_diff_community_after += 1
            if same_community_before:
                diff_class_same_community_before += 1
            else:
                diff_class_diff_community_before += 1
    fgap, _, _, _ = spectral_gap(newgraph)
    print()
    print(f"FinalGap = {fgap}") 
    data_modifying = end_algo - start_algo
    print(data_modifying)  
    newdata = from_networkx(newgraph)
    print(newdata)
    #data.edge_index = torch.cat([newdata.edge_index])  
    
    cluster_list = [cluster_dict_after[node] for node in range(len(data.y))]
    nmiscoremod = NMI(cluster_list, labels)
    
    return data, fgap, data_modifying, same_class_edges, diff_class_edges, nmiscoremod, same_class_same_community_after, same_class_diff_community_after, diff_class_same_community_after, diff_class_diff_community_after, same_class_same_community_before, same_class_diff_community_before, diff_class_same_community_before, diff_class_diff_community_before


def proxyaddmax(data, nxgraph, seed,max_iterations):
    print("Adding edges to maximize the gap...")
    
    
    # Track the original edges
    original_edges = set(nxgraph.edges())
    
    # Perform community detection before rewiring
    clustermod_before = maximize_modularity(nxgraph)
    cluster_dict_before = {node: i for i, cluster in enumerate(clustermod_before) for node in cluster}
    
    # Initialize counters before rewiring
    same_class_same_community_before = 0
    same_class_diff_community_before = 0
    diff_class_same_community_before = 0
    diff_class_diff_community_before = 0
    
    # Assuming `data.y` contains the node labels
    labels = data.y.cpu().numpy()
    
    # Count same-class and different-class edges before rewiring
    # for edge in original_edges:
    #     node1, node2 = edge
    #     same_class = labels[node1] == labels[node2]
    #     same_community = cluster_dict_before[node1] == cluster_dict_before[node2]
        
    #     if same_class:
    #         if same_community:
    #             same_class_same_community_before += 1
    #         else:
    #             same_class_diff_community_before += 1
    #     else:
    #         if same_community:
    #             diff_class_same_community_before += 1
    #         else:
    #             diff_class_diff_community_before += 1
    start_algo = time.time()
    newgraph = process_and_update_edges(nxgraph, rank_by_proxy_add, "proxyaddmax",seed, max_iter=max_iterations, updating_period=1)
    newgraph.remove_edges_from(list(nx.selfloop_edges(newgraph)))
    end_algo = time.time()
    # Track the edges after addition
    updated_edges = set(newgraph.edges())
    
    # Determine the added edges
    added_edges = updated_edges - original_edges
    
    # Initialize counters after rewiring
    same_class_edges = 0
    diff_class_edges = 0
    same_class_same_community_after = 0
    same_class_diff_community_after = 0
    diff_class_same_community_after = 0
    diff_class_diff_community_after = 0
    
    # Perform community detection after rewiring
    clustermod_after = maximize_modularity(newgraph)
    cluster_dict_after = {node: i for i, cluster in enumerate(clustermod_after) for node in cluster}
    
    # Count same-class and different-class edges after rewiring
    for edge in added_edges:
        node1, node2 = edge
        same_class = labels[node1] == labels[node2]

        same_community_before = cluster_dict_before[node1] == cluster_dict_before[node2]
        same_community_after = cluster_dict_after[node1] == cluster_dict_after[node2]
        if same_class:
            same_class_edges += 1
            if same_community_after:
                same_class_same_community_after += 1 
            else:
                same_class_diff_community_after += 1
            if same_community_before:
                same_class_same_community_before += 1
            else:
                same_class_diff_community_before += 1
        else:
            diff_class_edges += 1
            if same_community_after:
                diff_class_same_community_after += 1
            else:
                diff_class_diff_community_after += 1
            if same_community_before:
                diff_class_same_community_before += 1
            else:
                diff_class_diff_community_before += 1
        

    fgap, _, _, _ = spectral_gap(newgraph)
    print()
    print(f"FinalGap = {fgap}") 
    data_modifying = end_algo - start_algo
    print(data_modifying)  
    newdata = from_networkx(newgraph)
    print(newdata)
    #data.edge_index = torch.cat([newdata.edge_index])  
    
    cluster_list = [cluster_dict_after[node] for node in range(len(data.y))]
    nmiscoremod = NMI(cluster_list, labels)
    
    return data, fgap, data_modifying, same_class_edges, diff_class_edges, nmiscoremod, same_class_same_community_after, same_class_diff_community_after, diff_class_same_community_after, diff_class_diff_community_after, same_class_same_community_before, same_class_diff_community_before, diff_class_same_community_before, diff_class_diff_community_before


def proxyaddmin(data, nxgraph,seed, max_iterations):
    print("Adding edges to minimize the gap...")
    
    
    # Track the original edges
    original_edges = set(nxgraph.edges())
    
    # Perform community detection before rewiring
    clustermod_before = maximize_modularity(nxgraph)
    cluster_dict_before = {node: i for i, cluster in enumerate(clustermod_before) for node in cluster}
    
    # Initialize counters before rewiring
    same_class_same_community_before = 0
    same_class_diff_community_before = 0
    diff_class_same_community_before = 0
    diff_class_diff_community_before = 0
    
    # Assuming `data.y` contains the node labels
    labels = data.y.cpu().numpy()
    start_algo = time.time()
    newgraph = min_and_update_edges(nxgraph, rank_by_proxy_add_min, "proxyaddmin", seed,max_iter=max_iterations, updating_period=1)
    newgraph.remove_edges_from(list(nx.selfloop_edges(newgraph)))
    end_algo = time.time()
    # Track the edges after addition
    updated_edges = set(newgraph.edges())
    
    # Determine the added edges
    added_edges = updated_edges - original_edges
    
    # Initialize counters after rewiring
    same_class_edges = 0
    diff_class_edges = 0
    same_class_same_community_after = 0
    same_class_diff_community_after = 0
    diff_class_same_community_after = 0
    diff_class_diff_community_after = 0
    
    # Perform community detection after rewiring
    clustermod_after = maximize_modularity(newgraph)
    cluster_dict_after = {node: i for i, cluster in enumerate(clustermod_after) for node in cluster}
    
    # Count same-class and different-class edges after rewiring
    for edge in added_edges:
        node1, node2 = edge
        same_class = labels[node1] == labels[node2]
        same_community_before = cluster_dict_before[node1] == cluster_dict_before[node2]
        same_community_after = cluster_dict_after[node1] == cluster_dict_after[node2]
        if same_class:
            same_class_edges += 1
            if same_community_after:
                same_class_same_community_after += 1 
            else:
                same_class_diff_community_after += 1
            if same_community_before:
                same_class_same_community_before += 1
            else:
                same_class_diff_community_before += 1
        else:
            diff_class_edges += 1
            if same_community_after:
                diff_class_same_community_after += 1
            else:
                diff_class_diff_community_after += 1
            if same_community_before:
                diff_class_same_community_before += 1
            else:
                diff_class_diff_community_before += 1
        

    
    fgap, _, _, _ = spectral_gap(newgraph)
    print()
    print(f"FinalGap = {fgap}") 
    data_modifying = end_algo - start_algo
    print(data_modifying)  
    newdata = from_networkx(newgraph)
    print(newdata)
    #data.edge_index = torch.cat([newdata.edge_index])  
    
    cluster_list = [cluster_dict_after[node] for node in range(len(data.y))]
    nmiscoremod = NMI(cluster_list, labels)
    
    return data, fgap, data_modifying, same_class_edges, diff_class_edges, nmiscoremod, same_class_same_community_after, same_class_diff_community_after, diff_class_same_community_after, diff_class_diff_community_after, same_class_same_community_before, same_class_diff_community_before, diff_class_same_community_before, diff_class_diff_community_before


def fosr(data, max_iterations):
    print("Adding edges using FoSR...")
    
    # Convert to NetworkX graph
    nxgraph = to_networkx(data, to_undirected=True)
    
    # Track the original edges
    original_edges = set(nxgraph.edges())
    
    # Perform community detection before rewiring
    clustermod_before = maximize_modularity(nxgraph)
    cluster_dict_before = {node: i for i, cluster in enumerate(clustermod_before) for node in cluster}
    
    # Assuming `data.y` contains the node labels
    labels = data.y.cpu().numpy()
    
    start_algo = time.time()
    for j in tqdm(range(max_iterations)):
        edge_index, edge_type, _, prod = edge_rewire(data.edge_index.numpy(), num_iterations=1)      
        data.edge_index = torch.tensor(edge_index)
    data.edge_index = torch.cat([data.edge_index])
    end_algo = time.time()
    
    # Convert back to NetworkX graph after rewiring
    newgraph = to_networkx(data, to_undirected=True)
    
    # Track the edges after rewiring
    updated_edges = set(newgraph.edges())
    
    # Determine the added edges
    added_edges = updated_edges - original_edges
    
    # Initialize counters
    same_class_edges = 0
    diff_class_edges = 0
    same_class_same_community_after = 0
    same_class_diff_community_after = 0
    diff_class_same_community_after = 0
    diff_class_diff_community_after = 0
    same_class_same_community_before = 0
    same_class_diff_community_before = 0
    diff_class_same_community_before = 0
    diff_class_diff_community_before = 0
    
    # Perform community detection after rewiring
    clustermod_after = maximize_modularity(newgraph)
    cluster_dict_after = {node: i for i, cluster in enumerate(clustermod_after) for node in cluster}
    
    # Count same-class and different-class edges after rewiring
    for edge in added_edges:
        node1, node2 = edge
        same_class = labels[node1] == labels[node2]
        same_community_before = cluster_dict_before[node1] == cluster_dict_before[node2]
        same_community_after = cluster_dict_after[node1] == cluster_dict_after[node2]
        
        if same_class:
            same_class_edges += 1
            if same_community_after:
                same_class_same_community_after += 1 
            else:
                same_class_diff_community_after += 1
            if same_community_before:
                same_class_same_community_before += 1
            else:
                same_class_diff_community_before += 1
        else:
            diff_class_edges += 1
            if same_community_after:
                diff_class_same_community_after += 1
            else:
                diff_class_diff_community_after += 1
            if same_community_before:
                diff_class_same_community_before += 1
            else:
                diff_class_diff_community_before += 1
    
    fgap, _, _, _ = spectral_gap(newgraph)
    print(f"FinalGap = {fgap}")
    
    data_modifying = end_algo - start_algo
    print(f"Time taken: {data_modifying}")
    
    cluster_list = [cluster_dict_after[node] for node in range(len(data.y))]
    nmiscoremod = NMI(cluster_list, labels)
    
    return data, fgap, data_modifying, same_class_edges, diff_class_edges, nmiscoremod, same_class_same_community_after, same_class_diff_community_after, diff_class_same_community_after, diff_class_diff_community_after, same_class_same_community_before, same_class_diff_community_before, diff_class_same_community_before, diff_class_diff_community_before

def sdrf(data, max_iterations,removal_bound,tau):
          #print("Rewiring using SDRF...")
          start_algo = time.time()
          Newdatapyg = sdrf(data,max_iterations,removal_bound,tau)
          end_algo = time.time()
          data_modifying = end_algo - start_algo
          newgraph = to_networkx(Newdatapyg, to_undirected=True)
          fgap,_, _, _ = spectral_gap(newgraph)
          data = from_networkx(Newdatapyg)
          return data, fgap, data_modifying


def random_delete(data, seed,max_iterations):
    random.seed(seed)
    np.random.seed(seed)
    print("Deleting edges randomly...")
    graph = to_networkx(data, to_undirected=True)

    # Track the original edges
    original_edges = set(graph.edges())

    # Perform community detection before rewiring
    clustermod_before = maximize_modularity(graph)
    cluster_dict_before = {node: i for i, cluster in enumerate(clustermod_before) for node in cluster}

    # Initialize counters before rewiring
    same_class_same_community_before = 0
    same_class_diff_community_before = 0
    diff_class_same_community_before = 0
    diff_class_diff_community_before = 0

    # Assuming `data.y` contains the node labels
    labels = data.y.cpu().numpy()

    start_algo = time.time()
    if max_iterations > graph.number_of_edges():
        raise ValueError("Number of edges to delete exceeds the total number of edges in the graph.")  
    edges = list(graph.edges())
    edges_to_delete = random.sample(edges, max_iterations)   
    graph.remove_edges_from(edges_to_delete)
    end_algo = time.time()

    # Track the edges after deletion
    updated_edges = set(graph.edges())

    # Determine the deleted edges
    deleted_edges = original_edges - updated_edges

    # Initialize counters after rewiring
    same_class_edges = 0
    diff_class_edges = 0
    same_class_same_community_after = 0
    same_class_diff_community_after = 0
    diff_class_same_community_after = 0
    diff_class_diff_community_after = 0

    # Perform community detection after rewiring
    clustermod_after = maximize_modularity(graph)
    cluster_dict_after = {node: i for i, cluster in enumerate(clustermod_after) for node in cluster}

    # Count same-class and different-class edges after rewiring
    for edge in deleted_edges:
        node1, node2 = edge
        same_class = labels[node1] == labels[node2]
        same_community_before = cluster_dict_before[node1] == cluster_dict_before[node2]
        same_community_after = cluster_dict_after[node1] == cluster_dict_after[node2]
        if same_class:
            same_class_edges += 1
            if same_community_after:
                same_class_same_community_after += 1 
            else:
                same_class_diff_community_after += 1
            if same_community_before:
                same_class_same_community_before += 1
            else:
                same_class_diff_community_before += 1
        else:
            diff_class_edges += 1
            if same_community_after:
                diff_class_same_community_after += 1
            else:
                diff_class_diff_community_after += 1
            if same_community_before:
                diff_class_same_community_before += 1
            else:
                diff_class_diff_community_before += 1

    data_modifying = end_algo - start_algo
    newdata = from_networkx(graph)
    fgap, _, _, _ = spectral_gap(graph)
    print(f"FinalGap = {fgap}")
    print(f"Time taken: {data_modifying}")

    cluster_list = [cluster_dict_after[node] for node in range(len(data.y))]
    nmiscoremod = NMI(cluster_list, labels)

    return newdata, fgap, data_modifying, same_class_edges, diff_class_edges, nmiscoremod, same_class_same_community_after, same_class_diff_community_after, diff_class_same_community_after, diff_class_diff_community_after, same_class_same_community_before, same_class_diff_community_before, diff_class_same_community_before, diff_class_diff_community_before

def random_add(data, seed,max_iterations):
    random.seed(seed)
    np.random.seed(seed)
    print("Adding edges randomly...")
    graph = to_networkx(data, to_undirected=True)

    # Track the original edges
    original_edges = set(graph.edges())

    # Perform community detection before rewiring
    clustermod_before = maximize_modularity(graph)
    cluster_dict_before = {node: i for i, cluster in enumerate(clustermod_before) for node in cluster}

    # Initialize counters before rewiring
    same_class_same_community_before = 0
    same_class_diff_community_before = 0
    diff_class_same_community_before = 0
    diff_class_diff_community_before = 0

    # Assuming `data.y` contains the node labels
    labels = data.y.cpu().numpy()

    start_algo = time.time()
    nodes = list(graph.nodes())
    edges_added = 0
    while edges_added < max_iterations:
        node1, node2 = random.sample(nodes, 2)
        if node1 != node2 and not graph.has_edge(node1, node2):
            graph.add_edge(node1, node2)
            edges_added += 1
    end_algo = time.time()

    # Track the edges after addition
    updated_edges = set(graph.edges())

    # Determine the added edges
    added_edges = updated_edges - original_edges

    # Initialize counters after rewiring
    same_class_edges = 0
    diff_class_edges = 0
    same_class_same_community_after = 0
    same_class_diff_community_after = 0
    diff_class_same_community_after = 0
    diff_class_diff_community_after = 0

    # Perform community detection after rewiring
    clustermod_after = maximize_modularity(graph)
    cluster_dict_after = {node: i for i, cluster in enumerate(clustermod_after) for node in cluster}

    # Count same-class and different-class edges after rewiring
    for edge in added_edges:
        node1, node2 = edge
        same_class = labels[node1] == labels[node2]
        same_community_before = cluster_dict_before[node1] == cluster_dict_before[node2]
        same_community_after = cluster_dict_after[node1] == cluster_dict_after[node2]
        if same_class:
            same_class_edges += 1
            if same_community_after:
                same_class_same_community_after += 1 
            else:
                same_class_diff_community_after += 1
            if same_community_before:
                same_class_same_community_before += 1
            else:
                same_class_diff_community_before += 1
        else:
            diff_class_edges += 1
            if same_community_after:
                diff_class_same_community_after += 1
            else:
                diff_class_diff_community_after += 1
            if same_community_before:
                diff_class_same_community_before += 1
            else:
                diff_class_diff_community_before += 1

    data_modifying = end_algo - start_algo
    newdata = from_networkx(graph)
    fgap, _, _, _ = spectral_gap(graph)
    print(f"FinalGap = {fgap}")
    print(f"Time taken: {data_modifying}")

    cluster_list = [cluster_dict_after[node] for node in range(len(data.y))]
    nmiscoremod = NMI(cluster_list, labels)

    return newdata, fgap, data_modifying, same_class_edges, diff_class_edges, nmiscoremod, same_class_same_community_after, same_class_diff_community_after, diff_class_same_community_after, diff_class_diff_community_after, same_class_same_community_before, same_class_diff_community_before, diff_class_same_community_before, diff_class_diff_community_before
