import time
import networkx as nx
import scipy.sparse as sp
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from spectral_utils import *

rank_by_proxy_add = lambda g, gap, vecs,deg: rank_by(g, gap, vecs,deg, proxy_add_score, "add")
rank_by_proxy_delete = lambda g, gap, vecs,deg: rank_by(g, gap, vecs, deg,proxy_delete_score, "delete")
proxy_add_score = lambda g, edge, gap, vecs,deg: (gap_from_proxy(edge, gap, vecs,deg, 1), 1)
proxy_delete_score = lambda g, edge, gap, vecs,deg: (gap_from_proxy(edge, gap, vecs, deg,-1), -1)


def gap_from_proxy(edge, gap, vecs, deg,delta_w):
    """
    Approximately calculate the spectral gap of the graph after deleting the edge (i,j) via a proxy.
    """
    i, j = edge
    vecs = np.divide(vecs, np.sqrt(deg[:, np.newaxis]))
    vecs = vecs / np.linalg.norm(vecs, axis=1)[:, np.newaxis]
    return delta_w * ((vecs[i,1]-vecs[j,1])**2-gap*(vecs[i,1]**2 + vecs[j,1]**2))

def rank_by(g, gap, vecs, deg,score_method, add_or_delete):
    """
    Rank edges in the graph by the score_method (max).
    score_method returns a tuple (dgap, pm) where pm is 1 if adding and -1 if deleting
    """
    if add_or_delete == "add": 
        edges = list(nx.non_edges(g))
    elif add_or_delete == "delete": # all edges without self-loops
        edges = list(g.edges - nx.selfloop_edges(g))
    else:
        # all edges in or not in the graph or raise exception
        edges = list(nx.non_edges(g)) + list(g.edges - nx.selfloop_edges(g))
    
    edge_dgap_mapping = dict()
    for i, j in edges:
        edge_dgap_mapping[(i, j)] = score_method(g, (i, j), gap, vecs,deg)

    return sorted(edge_dgap_mapping.items(), key=lambda x: x[1][0], reverse=True)

def modify_k_edges(g, ranking_method, gap, vecs, deg, L_norm, k = 1):
    """
    Delete the edge with the maximum spectral gap of the graph after deleting the edge.
    """
    best_edges = ranking_method(g, gap, vecs,deg)
    counter = 0
    for i in range(len(best_edges)):
        (s, t), (dgap, pm) = best_edges[i]
        # print("{} edge".format("Adding" if pm == 1 else "Deleting"), (s, t), "\twith dgap", dgap)

        if pm == 1: 
            g.add_edge(s, t)
            new_gap, new_vecs, new_deg, new_Lnorm = spectral_gap_add(g, (s, t, deg, L_norm))
        else: 
            g.remove_edge(s, t)
            if not nx.is_connected(g):
                #print("Graph disconnected. Putting edge back", (s, t))
                g.add_edge(s, t)
                continue

            new_gap, new_vecs, new_deg, new_Lnorm = spectral_gap_delete(g, (s, t, deg, L_norm))
        counter += 1
        gap, vecs, deg, L_norm = new_gap, new_vecs, new_deg, new_Lnorm
        if counter == k:
            return True, g, gap, vecs, deg, L_norm
    
    print("No more edges can be modified to increase the spectral gap.")
    return False, g, gap, vecs, deg, L_norm


def process_and_update_edges(g, ranking_method, ranking_name, updating_period=1, max_iter=np.inf):
    """
    Process and update all edges in the graph
    according to the maximum spectral gap of the graph after deleting the edge;
    not calculated directly but using the proxy delete method.
    """
    e0 = g.number_of_edges()
    g = add_self_loops(g) 
    start = time.time()

    gap, vecs, deg, L_norm = spectral_gap(g)
    print(deg)
    print("=" * 40)

    # gaps = [gap]
    counter = 0
    modified = True
    with tqdm(total=max_iter, desc="Edge Modification") as pbar:
        while modified and counter < max_iter : #len(gaps) <= max_iter:
            modified, g, gap, vecs, deg,L_norm = modify_k_edges(g, ranking_method, gap, vecs, deg, L_norm, updating_period)
            #modified = modify_k_edges(g, ranking_method, gap, vecs, deg, L_norm, updating_period)
            counter+=1 
            pbar.update(1)  # Update the progress bar
            if len(g.edges) == 0 or not modified:
              break
    e1 = g.number_of_edges() - g.number_of_nodes()


    save_gaps(time.time()-start,
                    updating_period,
                    (e0, e1),
                    "{}.csv".format(ranking_name))
    return g

def save_gaps(seconds, k, edge_change, path):
    """
    Save the gaps sequence to a file as a csv file.
    """
    e0, e1 = edge_change
    with open(path, "w") as f:
        f.write("seconds, k, e0, e1\n")
        f.write("{}, {}, {}, {}\n".format(seconds, k, e0, e1))