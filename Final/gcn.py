import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from argparse import ArgumentParser
import pickle
import os.path
import networkx as nx
import data_utils as utils
from sklearn.preprocessing import MinMaxScaler


########################
# Function Definitions #
########################

"""
Getting the Data
"""

def is_valid_file(parser, arg):
    """
    :param parser: parser object that was initialized
    :param arg: io.TextIOWrapper
    :return: return an open file handle
    """
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return open(arg, 'rb')  # return an open file handle


"""
Get number of classes
"""

def get_node_class_list(graphs):
    """
    :param graphs: networkx list of graphs
    :return: list of target classes
    """
    target_class = set()
    for graph in graphs:
        labels = utils.get_node_labels(nx.Graph(graph))
        target_class.update(labels)
    return target_class



"""
# Get symmetric normalized matrix
"""

def get_symmetric_normalized(A):
    """
    :param A: adjacency matrix
    :return: normalized A_tilda matrix
    """
    
    """
    # Sparse version
    m, n = A.shape
    id = sparse.eye(m,n,0,dtype=float)
    A_hat = (A.tocsr() + id.tocsr()).tolil()
    D = A_hat.sum(axis=0)
    D_diag = sparse.diags(D.A1)
    """
    
    m, n = np.shape(A)
    A_hat = A + np.eye(m, n)
    D = A_hat.sum(axis=0)
    D_diag = np.diag(D)
    D_diag_inv = np.linalg.solve(D_diag, np.eye(m, n))
    D_diag_inv_sqrt = np.power(D_diag_inv, 0.5)
    A_temp = np.matmul(D_diag_inv_sqrt, A_hat)
    A_tilda = np.matmul(A_temp, D_diag_inv_sqrt)
    return A_tilda



"""
Get maximum attribute length
"""

def get_max_attr_length(graph):
    """
    :param graph: networkx list of graphs
    :return: maximal length of the node attributes vector
    """
    length = set()
    graph = nx.Graph(graph[0])
    nodes = graph.nodes
    for data in nodes.data():
        attributes = data[1]['node_attributes']
        l = len(attributes)
        length.update([l])

    return max(length)



"""
# Get maximum attribute length
"""
def get_num_nodes(graph):
    """
    :param graphs: networkx list of graphs
    :return: total number of ndoes
    """
    graph = nx.Graph(graph[0])
    data = graph.nodes.data()
    num_nodes = np.shape(data)[0]
    return num_nodes



"""
# Get attribute matrix
"""
def get_attribute_matrix(graph):
    """
    :param graph: networkx graph object
    :return: |V|xk matrix where k = |attribute vector|
    """
    G = nx.Graph(graph)
    data = G.nodes.data()
    num_nodes = np.shape(data)[0]
    # Assumption - constant length of attributes
    num_attributes = np.array(data[1]['node_attributes']).shape[0]

    attributes = np.zeros((num_nodes, num_attributes), dtype=np.float32)
    index = 0
    for entry in data:
        # version that contains node labels as well
        # arr = np.append(entry[1]['node_attribute'], entry[1]['node_label'])
        # version that only contains attributes
        arr = entry[1]['node_attributes']
        attributes[index] = arr
        index += 1

    return attributes



"""
# Get node labels
"""
def get_node_labels(graph):
    """
    :param graph: networkx graph object
    :return: |V| vector containing node labels
    """
    G = nx.Graph(graph)
    data = G.nodes.data()
    num_nodes = np.shape(data)[0]

    # labels = tf.zeros(num_nodes)
    index = 0
    # labels_list = []
    labels = np.zeros(num_nodes)
    for entry in data:
        labels[index] = entry[1]['node_label']
        index += 1

    return labels


########################
### Class Definitions ###
########################

"""
Class definition of convolution layer
"""
class NodeConv(keras.layers.Layer):
    def __init__(self, num_classes, n_output_nodes, activation="relu", **kwargs):
        """
        :param num_classes: number of target classes
        :param n_output_nodes: number of output nodes
        :param activation: activation function
        """
        super(NodeConv, self).__init__(**kwargs)
        self.n_output_nodes = n_output_nodes
        self.num_classes = num_classes
        self.activation = keras.layers.Activation(activation)

    def build(self, input_shape):
        """
        :param input_shape: input tensor shape
        :return: NA
        """
        a_shape, x_shape = input_shape
        self.num_vertices = a_shape[1]
        w_init = keras.initializers.HeNormal(seed=None)
        self.weight = tf.Variable(
            initial_value=w_init(shape=(x_shape[2], self.n_output_nodes),
                                 dtype='float32'), trainable=True)

    def call(self, inputs):
        """
        :param inputs: input tensor shape
        :return: output of a particular node
        """
        a_tilda, x = inputs[0], inputs[1]
        y = tf.tensordot(tf.matmul(a_tilda, x), self.weight, axes=[2, 0])
        x_next = self.activation(y)
        return x_next


"""
Class definition of pooling layer
"""
class PoolingLayer(tf.keras.layers.Layer):
    def __init__(self, name="PoolingLayer", **kwargs):
        super(PoolingLayer, self).__init__(name=name, **kwargs)

    def call(self, inputs):
        """
        Sums the node embedding of each graph element-wise
        :param inputs: tensor that represent
        :return:
        """
        x = inputs
        pool = tf.reduce_sum(x, axis=1)
        return pool
