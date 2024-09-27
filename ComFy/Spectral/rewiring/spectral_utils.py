import networkx as nx
import scipy.sparse as sp
import numpy as np
eps = 1e-6

def add_self_loops(g):
    """
    Add self-loops to the graph. Returns modified graph in graph format.
    """
    g.add_edges_from([(i,i) for i in range(len(g.nodes))])
    return g

def obtain_Lnorm(g):
    """
    Obtain the normalized Laplacian matrix of the graph,
    as well as the degree of each node.
    Assumes self-loops in the graph.
    """
    adj = nx.adjacency_matrix(g)

    # get degree matrix
    deg = np.array(adj.sum(axis=1)).flatten()

    D_sqrt_inv = sp.diags(np.power(deg+eps, -0.5))
    L_norm = (sp.eye(adj.shape[0]) - D_sqrt_inv @ adj @ D_sqrt_inv)
    return deg, L_norm

def update_Lnorm_deletion(u, v, L_norm, deg):
    """
    Update the normalized Laplacian matrix of the graph,
    as well as the degree of each node,
    after deleting the edge (u,v).
    """
    L_norm[u, :] *= np.sqrt(deg[u] / (deg[u] - 1))
    L_norm[:, u] = L_norm[u, :].T
    L_norm[v, :] *= np.sqrt(deg[v] / (deg[v] - 1))
    L_norm[:, v] = L_norm[v, :].T
    deg[u] -= 1
    deg[v] -= 1
    L_norm[u,u] = 1-1/deg[u]
    L_norm[v,v] = 1-1/deg[v]
    L_norm[u, v] = 0
    L_norm[v, u] = 0
    return deg, L_norm



def update_Lnorm_addition(u,v,L_norm,deg):
      L_norm[u, :] *= np.sqrt(deg[u] / (deg[u] + 1))
      L_norm[:, u] = L_norm[u, :].T
      L_norm[v,:] *= np.sqrt(deg[v]/(deg[v] + 1))
      L_norm[:,v] = L_norm[v,:].T
      L_norm[u,v] = -1/np.sqrt((deg[u]+1)*(deg[v]+1))
      L_norm[v,u] = L_norm[u,v] 
      deg[u]+=1
      deg[v]+=1
      L_norm[u,u] = 1-1/deg[u]
      L_norm[v,v] = 1-1/deg[v]
      return deg,L_norm

def spectral_gap(g, params=None):
    """
    Calculate the spectral gap of the graph.
    """

    deg, L_norm = obtain_Lnorm(g)
    try: # use sparse eigsh if possible
        vals, vecs = sp.linalg.eigsh(L_norm, k=2,sigma=0.0,which='LM')
        vecs = np.divide(vecs, np.sqrt(deg[:, np.newaxis]))
        vecs = vecs / np.linalg.norm(vecs, axis=1)[:, np.newaxis]
    except: # use dense eig if Factor is exactly singular
        dense_Lnorm = nx.normalized_laplacian_matrix(g).todense()
        vals, vecs = np.linalg.eigh(dense_Lnorm)
        vecs = np.divide(vecs, np.sqrt(deg[:, np.newaxis]))
        vecs = vecs / np.linalg.norm(vecs, axis=1)[:, np.newaxis]
    
    return vals[1], vecs, deg, L_norm


def spectral_gap_add(g, params=None):
    """
    Calculate the spectral gap of the graph.
    """
    if params is None: 
        deg, L_norm = obtain_Lnorm(g)
    else: 
        u, v, deg, L_norm = params
        deg, L_norm = update_Lnorm_addition(u, v, L_norm, deg)

    try: # use sparse eigsh if possible
        vals, vecs = sp.linalg.eigsh(L_norm, k=2,sigma=0.0,which='LM')
        vecs = np.divide(vecs, np.sqrt(deg[:, np.newaxis]))
        vecs = vecs / np.linalg.norm(vecs, axis=1)[:, np.newaxis]
    except: # use dense eig if Factor is exactly singular
        dense_Lnorm = nx.normalized_laplacian_matrix(g).todense()
        vals, vecs = np.linalg.eigh(dense_Lnorm)
        vecs = np.divide(vecs, np.sqrt(deg[:, np.newaxis]))
        vecs = vecs / np.linalg.norm(vecs, axis=1)[:, np.newaxis]
    
    return vals[1], vecs, deg, L_norm



def spectral_gap_delete(g, params=None):
    """
    Calculate the spectral gap of the graph.
    """
    if params is None: 
        deg, L_norm = obtain_Lnorm(g)
    else: 
        u, v, deg, L_norm = params
        deg, L_norm = update_Lnorm_deletion(u, v, L_norm, deg)

    try: # use sparse eigsh if possible
        vals, vecs = sp.linalg.eigsh(L_norm, k=2,sigma=0.0,which='LM')
        vecs = np.divide(vecs, np.sqrt(deg[:, np.newaxis]))
        vecs = vecs / np.linalg.norm(vecs, axis=1)[:, np.newaxis]
    except: # use dense eig if Factor is exactly singular
        dense_Lnorm = nx.normalized_laplacian_matrix(g).todense()
        vals, vecs = np.linalg.eigh(dense_Lnorm)
        vecs = np.divide(vecs, np.sqrt(deg[:, np.newaxis]))
        vecs = vecs / np.linalg.norm(vecs, axis=1)[:, np.newaxis]
    
    return vals[1], vecs, deg, L_norm