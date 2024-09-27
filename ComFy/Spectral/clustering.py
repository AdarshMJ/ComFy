import networkx as nx
import numpy as np
import scipy.sparse.linalg
from sklearn.cluster import KMeans
import sklearn.preprocessing


def k_way_spectral(G, k):
        assert nx.is_connected(G), "the graph must be connnected"
        clusters = []
        if G.order() < k:
                clusters = list(G.nodes())
        else:
            L = nx.laplacian_matrix(G)
            _, eigenvecs = scipy.sparse.linalg.eigsh(L.asfptype(), k=k+1, which='SM')
            eigenvecs = eigenvecs[:, 1:]
            eigenvecs = sklearn.preprocessing.normalize(eigenvecs)
            kmeans = KMeans(n_clusters=k).fit(eigenvecs)
            cluster_labels = kmeans.labels_
            clusters = [[] for _ in range(max(cluster_labels) + 1)]
            for node_id, cluster_id in zip(G.nodes(), cluster_labels):
                            clusters[cluster_id].append(node_id)
        return clusters,cluster_labels

def maximize_modularity(G):
  return nx.community.greedy_modularity_communities(G)