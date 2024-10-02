import time
import networkx as nx
import scipy.sparse as sp
import numpy as np
from tqdm import tqdm
import random
import warnings
warnings.filterwarnings('ignore')

from rewiring.spectral_utils import *

rank_by_proxy_add_min = lambda g, gap, vecs: rank_by(g, gap, vecs, proxy_add_score, "add")
rank_by_proxy_delete_min = lambda g, gap, vecs: rank_by(g, gap, vecs, proxy_delete_score, "delete")

rank_by_greedy_add = lambda g, gap, vecs: rank_by(g, gap, vecs, sgap_add_score, "add")
rank_by_greedy_delete = lambda g, gap, vecs: rank_by(g, gap, vecs, sgap_delete_score, "delete")

proxy_add_score = lambda g, edge, gap, vecs: (gap_from_proxy(edge, gap, vecs, 1), 1)
proxy_delete_score = lambda g, edge, gap, vecs: (gap_from_proxy(edge, gap, vecs, -1), -1)
sgap_add_score = lambda g, edge, gap, vecs: (gap_from_spectral(g, edge, True), 1)
sgap_delete_score = lambda g, edge, gap, vecs: (gap_from_spectral(g, edge, False), -1)

def gap_from_proxy(edge, gap, vecs, delta_w):
    """
    Approximately calculate the spectral gap of the graph after deleting the edge (i,j) via a proxy.
    """
    i, j = edge
    return delta_w * ((vecs[i,1]-vecs[j,1])**2-gap*(vecs[i,1]**2 + vecs[j,1]**2))

def gap_from_spectral(g, edge, adding=True):
    """
    Calculate the spectral gap of the graph after deleting the edge (i,j) directly.
    """
    i, j = edge
    if adding: g.add_edge(i, j)
    else: g.remove_edge(i, j)

    gap, vecs, deg, L_norm = spectral_gap(g)

    if adding: g.remove_edge(i, j)
    else: g.add_edge(i, j)
    return gap

def rank_by(g, gap, vecs, score_method, add_or_delete):
    """
    Rank edges in the graph by the score_method (max).
    score_method returns a tuple (dgap, pm) where pm is 1 if adding and -1 if deleting
    """
    if add_or_delete == "add": 
        edges = list(nx.non_edges(g))
        edges = random.sample(edges,1500)
    elif add_or_delete == "delete": # all edges without self-loops
        edges = list(g.edges - nx.selfloop_edges(g))
    else:
        # all edges in or not in the graph or raise exception
        edges = list(nx.non_edges(g)) + list(g.edges - nx.selfloop_edges(g))
    
    edge_dgap_mapping = dict()
    for i, j in edges:
        edge_dgap_mapping[(i, j)] = score_method(g, (i, j), gap, vecs)

    # return sorted(edge_dgap_mapping.items(), key=lambda x: x[1][0], reverse=True)
    return list(edge_dgap_mapping.items())

def modify_k_edges(g, ranking_method, gap, vecs, deg, L_norm, k = 1):
    """
    Delete the edge with the maximum spectral gap of the graph after deleting the edge.
    """
    best_edges = ranking_method(g, gap, vecs)
    counter = 0
    for _ in range(len(best_edges)):
        (s, t), (dgap, pm) = min(best_edges, key=lambda x: x[1][0])
        best_edges.remove(((s, t), (dgap, pm)))
        # (s, t), (dgap, pm) = best_edges[i]

        if pm == 1: 
            g.add_edge(s, t)
            deg, L_norm = update_Lnorm_addition(s, t, L_norm, deg)
            
        else: 
            g.remove_edge(s, t)
            if not nx.is_connected(g):
                g.add_edge(s, t)
                continue
            deg, L_norm = update_Lnorm_deletion(s, t, L_norm, deg)

        # if new_gap <= gap:
        #     if pm == 1: g.remove_edge(s, t)
        #     else: g.add_edge(s, t)
        #     # print("Backtracking on", (s, t))
        #     continue

        counter += 1
        if counter == k:
            return True, g, deg, L_norm
    
    print("No more edges can be modified to increase the spectral gap.")
    return False, g, deg, L_norm


def min_and_update_edges(g, ranking_method, ranking_name, updating_period=1, max_iter=np.inf):
    """
    Process and update all edges in the graph
    according to the maximum spectral gap of the graph after deleting the edge;
    not calculated directly but using the proxy delete method.
    """
    e0 = g.number_of_edges()
    g = add_self_loops(g)
    start = time.time()

    deg, L_norm = obtain_Lnorm(g)
    gap, vecs,_,_ = spectral_gap(g)
    #print("Initial spectral gap:", gap)
    print("=" * 40)

    # gaps = [gap]
    counter = 0
    modified = True
    with tqdm(total=max_iter, desc="Edge Modification") as pbar:
        while modified and counter < max_iter : #len(gaps) <= max_iter:
            modified, g, deg, L_norm = modify_k_edges(g, ranking_method, gap, vecs, deg, L_norm, updating_period)
            gap, vecs,_,_ = spectral_gap(g)
            counter+=1 
            # gaps.append(gap)
            pbar.update(1)  # Update the progress bar
            if len(g.edges) == 0 or not modified:
              break
        # get number of nodes and edges
    e1 = g.number_of_edges() - g.number_of_nodes()
    # save_gaps(gaps, 
    #                 time.time()-start,
    #                 updating_period,
    #                 (e0, e1),
    #                 "{}.csv".format(ranking_name))
    # print("Final gap_holder:", gaps)

    # save_gaps(time.time()-start,
    #                 updating_period,
    #                 (e0, e1),
    #                 "{}.csv".format(ranking_name))
    return g

# def save_gaps(seconds, k, edge_change, path):
#     """
#     Save the gaps sequence to a file as a csv file.
#     """
#     e0, e1 = edge_change
#     with open(path, "w") as f:
#         f.write("seconds, k, e0, e1\n")
#         f.write("{}, {}, {}, {}\n".format(seconds, k, e0, e1))