import networkx as nx
import numpy as np
import tensorflow as tf
import itertools
from data_utils import get_node_labels, get_graph_label
from data_utils import get_padded_adjacency, get_padded_node_labels, get_padded_node_attributes


"""Useful functions"""

def get_edge_attributes(G):
    """
    :param G: A networkx graph G=(V,E)
    :return: A numpy array of shape (|E|, a), where a is the length of the edges attribute vector
    """
    attributes = np.array([e[2]['edge_attributes'] for e in list(G.edges(data=True))])
    return attributes



def get_padded_edge_attributes(graphs):
    """
    Computes a 3D Tensor X with shape (k, n, a) that stacks the node attributes of all graphs.
    Here, k = |graphs|, n = max(|E|) and a is the length of the attribute vectors.

    :param graphs: A list of networkx graphs
    :return: Numpy array X
    """
    edge_attributes = [get_edge_attributes(g) for g in graphs]

    max_size = np.max([nx.number_of_edges(g) for g in graphs])
    padded = [np.vstack([x, np.zeros((max_size-x.shape[0], x.shape[1]), dtype=np.float32)]) for x in edge_attributes]
    stacked = np.stack(padded, axis=0)
    return stacked




def double(M):
    """
    For doubling the dimensions
    M: a 3-D tensor
    :return the edge feature 
    """
    l = M.shape[0]
    ne = M.shape[1] 
    dm = np.zeros(shape=(l, 2*ne,M.shape[2]))
    for i in range(l):
        for j in range(ne):
            dm[i][2*j] = M[i][j]
            dm[i][2*j+1] = M[i][j]
    return dm




def incidence_generate(G):
    """
    Function that generates incidence matrices I_in, I_out for a given graph G

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
    
    return incidence_out, incidence_in



    
def get_maximums(train_set, test_set):
    """
    Function that calculates the maximum number of nodes and number of edges
    :param train_set: training set
    :param test_set: evaluation set
    :return: maximum edges, maximum nodes, edge classes present in dataset
    """
    # Set to hold number of nodes and edges in graphs
    edge = set()
    node = set()

    # Find number of edges and nodes across the dataset (including test and validate set)
    for graph in itertools.chain(train_set, test_set):
        num_nodes = len(list(graph.nodes()))                   # Number of nodes
        num_edges = len(list(graph.edges()))                   # Number of edges
        
        edge.update([num_edges])
        node.update([num_nodes])

    # Maximum number of edges and nodes
    max_num_edge = max(edge)
    max_num_node = max(node)

    return max_num_edge, max_num_node




def get_padded(graphs, max_ne, max_nn,weighted=False):
    """
    For padding incidence matrices I_out and I_in, to achieve a uniform size across the graphs dataset

    :param graphs: list of graphs
    :param weighted: boolean parameter for weighted version of I_in (default: False)
    :param max_nn: maximum number of nodes
    :param max_ne: maximum number of edges (undirected)
    :return: I_out: incidence matrix
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
   
    for i in range(l):
        result = incidence_generate(graphs[i])
        # update incidence out and incidence in for current graph
        I_out[i][:ne[i] * 2, :nn[i]] = result[0]
        I_in[i][:ne[i] * 2, :nn[i]] = result[1]

    return I_out, I_in



"""
Classes of Layers 
"""


class Linear(tf.keras.layers.Layer):
    """ Basic perceptron layer without bias"""
    
    def __init__(self, units=32):
        """
        :param units: the number of output units (default: 32)"""
        super(Linear, self).__init__()
        self.units = units
        
    def build(self, input_shape):
        """
        :param input_shape: input tensor shape
        """
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer="random_normal", trainable=True, )
    def call(self, inputs):
        return tf.matmul(inputs, self.w)



class MPGNN(tf.keras.layers.Layer):
    """MPGNN layer"""
    
    def __init__(self, units, res=False):
        """
        :param units: the number of output units
        :param res: boolean value for residual connections (default: False)
        """
        super(MPGNN, self).__init__()
        self.units = units
        self.res = res
        self.Mlayer = Linear(self.units)
        self.Ulayer = Linear(self.units)

    def build(self, input_shape):
        """
        :param input_shape: input tensor shape
        :return: NA
        """
        node_shape = input_shape[0]  # Shape of node embedding
        # Define W matrix for residual connections
        self.W = self.add_weight(shape=(node_shape[-1], self.units), initializer="random_normal", trainable=True)

    def call(self, inputs):
        """
        :param inputs: input tensor shape
        :return: updated node embeddings X
        """
        node_emd, edge_fea, I_out, I_in = inputs

        x1 = tf.matmul(I_out, node_emd)
        x1 = tf.keras.layers.Concatenate(axis=2)([edge_fea, x1])
        x1 = self.Mlayer(x1)
        S = tf.nn.relu(x1)

        x2 = tf.transpose(I_in, perm=[0, 2, 1])
        x2 = tf.matmul(x2, S)
        x2 = tf.keras.layers.Concatenate(axis=2)([node_emd, x2])
        x2 = self.Ulayer(x2)
        X = tf.nn.relu(x2)
        # Residual connections
        if not self.res == False:
            return X
        else:
            if node_emd.shape[-1] == self.units:
                return node_emd + X
            else:
                return tf.matmul(node_emd, self.W) + X



class PoolingLayer(tf.keras.layers.Layer):
    """Pooling Layer"""

    def __init__(self, name="PoolingLayer", **kwargs):
        super(PoolingLayer, self).__init__(name=name, **kwargs)

    def call(self, inputs):
        """
        :param inputs: input tensor shape
        :return: Sum Pooling layer 
        """
        x = inputs
        pool = tf.reduce_sum(x, axis=1)
        return pool
