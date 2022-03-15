#!/usr/bin/env python
# coding: utf-8

import networkx as nx
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import itertools

from data_utils import get_node_labels, get_graph_label
from data_utils import get_padded_adjacency, get_padded_node_labels, get_padded_node_attributes


def get_edge_label(G):
    """
    Function that encodes the edge label with one-hot-encoding scheme.

    :param G: networkx graph
    :return: one-hot vectors of edge labels 
    """
    onehot_encoder = OneHotEncoder(sparse=False)
    label = []
    edges = list(G.edges())  # List of edges
    for edge in edges:
        n1 = edge[0]
        n2 = edge[1]

        # Duplicate labels while converting undirected to directed
        label.append(G[n1][n2]['edge_label'])
        label.append(G[n1][n2]['edge_label'])
    label = np.array(label)
    label = label.reshape(len(label), 1)
    label = onehot_encoder.fit_transform(label)

    return label


def incidence_generate(G, weighted=False):
    """
    Function that generates incidence matrices for a given graph G

    :param G: networkx graph
    :param weighted: boolean value for weighted version of I_in (default: False)
    :return: incidence matrices I_out, I_in or weighted I_in of size (|Directed Edges| x |Node|)
    """

    nnodes = nx.number_of_nodes(G)  # Number of nodes
    nedges = nx.number_of_edges(G)  # Number of edges
    edges = list(G.edges())  # List of edges

    incidence_in = np.zeros(shape=(2 * nedges, nnodes))
    incidence_out = np.zeros(shape=(2 * nedges, nnodes))

    edge_index = 0
    for edge in edges:
        node_j = edge[0]
        node_k = edge[1]

        # Update for edge going from node_j to node_k
        incidence_out[edge_index][node_j] = 1
        incidence_in[edge_index][node_k] = 1

        # Update for edge going from node_k to node_j
        incidence_out[edge_index + 1][node_k] = 1
        incidence_in[edge_index + 1][node_j] = 1

        # Update the edge_index for next pair of nodes
        # Edge is undirected; need room for 2 edges for each undirected edge
        edge_index += 2

    # Calculate the weighted incidence matrix
    degree_array = np.sum(incidence_in, axis=0)
    degree_array = 1 / degree_array
    weighted_in = np.multiply(degree_array, incidence_in)

    if not weighted:
        return incidence_out, incidence_in, nnodes, nedges
    else:
        return incidence_out, weighted_in, nnodes, nedges


def get_maximums(train_set, valid_set, test_set):
    """
    Function that calculates the maximum number of nodes and number of edges

    :param train_set: training set
    :param valid_set: validation set
    :param test_set: test set
    :return: maximum edges, maximum nodes, edge classes present in dataset, node classes present in dataset
    """
    # Set to hold number of nodes and edges in graphs
    edge = set()
    node = set()
    edge_labels = set()
    node_labels = set()

    # Find number of edges and nodes across the dataset (including test and validate set)
    for graph in itertools.chain(train_set, valid_set, test_set):
        num_nodes = len(list(graph.nodes()))  # Number of nodes
        num_edges = len(list(graph.edges()))  # Number of edges
        elabels = nx.get_edge_attributes(graph, 'edge_label')  # Get edge attributes
        nlabels = get_node_labels(graph)  # Get node labels

        node_labels.update(nlabels)
        edge_labels.update(elabels.values())
        edge.update([num_edges])
        node.update([num_nodes])

    # List containing edge classes
    edge_labels = list(edge_labels)
    node_labels = list(node_labels)

    # Maximum number of edges and nodes
    max_num_edge = max(edge)
    max_num_node = max(node)

    return max_num_edge, max_num_node, edge_labels, node_labels


def get_padded(graphs, max_nn, max_ne, edge_labels, node_labels, weighted=False):
    """
    Function which returns the pad of node_feature, edge_feature and incidence matrices I_out and I_in,
    to achieve a uniform size across the graphs dataset

    :param graphs: list of graphs
    :param weighted: boolean parameter for weighted version of I_in (default: False)
    :param max_nn: maximum number of nodes
    :param max_ne: maximum number of edges (undirected)
    :param edge_labels: list of edge labels present in the dataset
    :param node_labels: list of node labels present in the dataset
    :return: node_feature
             edge_feature
             I_out: incidence matrix
             I_in: incidence matrix
    """

    l = len(graphs)  # Length of graph list
    nn = [nx.number_of_nodes(G) for G in graphs]  # Store the number of nodes of each G in graphs
    ne = [nx.number_of_edges(G) for G in graphs]  # Store the number of edges of each G in graphs

    # directed edges should be twice the number of undirected edges
    max_nde = 2 * max_ne

    # initialize storage 3d matrices with zeros
    I_out = np.zeros(shape=(l, max_nde, max_nn))  # Incidence_out matrix
    I_in = np.zeros(shape=(l, max_nde, max_nn))  # Incidence_in matrix
    edge_features = np.zeros(shape=(l, max_nde, len(edge_labels)))  # Edge features matrix

    nodes = get_padded_node_labels(graphs)
    node_features = np.zeros(shape=(l, max_nn, len(node_labels)))  # Node feature matrix

    for i in range(l):
        result = incidence_generate(graphs[i], weighted)

        # update incidence out and incidence in for current graph
        I_out[i][:ne[i] * 2, :nn[i]] = result[0]
        I_in[i][:ne[i] * 2, :nn[i]] = result[1]

        # get edge labels and update the edge features
        f = get_edge_label(graphs[i])
        edge_features[i][:ne[i] * 2, :f.shape[1]] = f

        # get nodes and update the node features
        n = nodes[i]
        node_features[i][:n.shape[0], :n.shape[1]] = n

    return node_features, edge_features, I_out, I_in
