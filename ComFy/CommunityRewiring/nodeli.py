
import numpy as np
import networkx as nx

#### Informativeness measures taken from - 
###Characterizing Graph Datasets for Node Classification: Homophily-Heterophily Dichotomy and Beyond, Platonov et al.###



def get_graph_and_labels_from_pyg_dataset(dataset):
    graph = nx.Graph()
    graph.add_nodes_from(range(len(dataset.x)))
    graph.add_edges_from(dataset.edge_index.T.numpy())

    labels = dataset.y.detach().cpu().numpy()
    return graph, labels


def convert_labels_to_consecutive_integers(labels):
    unique_labels = np.unique(labels)
    labels_map = {label: i for i, label in enumerate(unique_labels)}
    new_labels = np.array([labels_map[label] for label in labels])

    return new_labels


def li_node(graph, labels, eps=1e-8):
    """Compute node label informativeness."""
    labels = convert_labels_to_consecutive_integers(labels)

    num_classes = len(np.unique(labels))

    class_probs = np.array([0 for _ in range(num_classes)], dtype=float)
    class_degree_weighted_probs = np.array([0 for _ in range(num_classes)], dtype=float)
    num_zero_degree_nodes = 0
    for u in graph.nodes:
        if graph.degree(u) == 0:
            num_zero_degree_nodes += 1
            continue

        label = labels[u]
        class_probs[label] += 1
        class_degree_weighted_probs[label] += graph.degree(u)

    class_probs /= class_probs.sum()
    class_degree_weighted_probs /= class_degree_weighted_probs.sum()
    num_nonzero_degree_nodes = len(graph.nodes) - num_zero_degree_nodes


    edge_probs = np.zeros((num_classes, num_classes))
    for u, v in graph.edges:
        label_u = labels[u]
        label_v = labels[v]
        edge_probs[label_u, label_v] += 1 / (num_nonzero_degree_nodes * graph.degree(u))
        edge_probs[label_v, label_u] += 1 / (num_nonzero_degree_nodes * graph.degree(v))

    edge_probs += eps

    log = np.log(edge_probs / (class_probs.reshape(-1, 1) * class_degree_weighted_probs.reshape(1, -1)))
    numerator = (edge_probs * log).sum()
    denominator = (class_probs * np.log(class_probs)).sum()
    li_node = - numerator / denominator

    return li_node

def h_edge(graph, labels):
    """Compute edge homophily."""
    edges_with_same_label = 0
    for u, v in graph.edges:
        if labels[u] == labels[v]:
            edges_with_same_label += 1

    h_edge = edges_with_same_label / len(graph.edges)

    return h_edge


def h_adj(graph, labels):
    """Compute adjusted homophily."""
    labels = convert_labels_to_consecutive_integers(labels)

    num_classes = len(np.unique(labels))

    degree_sums = np.zeros((num_classes,))
    for u in graph.nodes:
        label = labels[u]
        degree_sums[label] += graph.degree(u)

    adjust = (degree_sums ** 2 / (len(graph.edges) * 2) ** 2).sum()

    h_adj = (h_edge(graph, labels) - adjust) / (1 - adjust)

    return h_adj


def li_edge(graph, labels, eps=1e-8):
    """Compute edge label informativeness."""
    labels = convert_labels_to_consecutive_integers(labels)

    num_classes = len(np.unique(labels))

    class_probs = np.array([0 for _ in range(num_classes)], dtype=float)
    class_degree_weighted_probs = np.array([0 for _ in range(num_classes)], dtype=float)
    for u in graph.nodes:
        label = labels[u]
        class_probs[label] += 1
        class_degree_weighted_probs[label] += graph.degree(u)

    class_probs /= class_probs.sum()
    class_degree_weighted_probs /= class_degree_weighted_probs.sum()

    edge_probs = np.zeros((num_classes, num_classes))
    for u, v in graph.edges:
        label_u = labels[u]
        label_v = labels[v]
        edge_probs[label_u, label_v] += 1
        edge_probs[label_v, label_u] += 1

    edge_probs /= edge_probs.sum()

    edge_probs += eps

    numerator = (edge_probs * np.log(edge_probs)).sum()
    denominator = (class_degree_weighted_probs * np.log(class_degree_weighted_probs)).sum()
    li_edge = 2 - numerator / denominator

    return li_edge
