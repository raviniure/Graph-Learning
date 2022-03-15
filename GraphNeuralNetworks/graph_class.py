#!/usr/bin/env python
# coding: utf-8

import os
from data_utils import get_node_labels
from data_utils import get_graph_label
from data_utils import get_padded_node_labels
from data_utils import get_node_attributes
from data_utils import get_padded_node_attributes
import data_utils as utils
import networkx as nx
import tensorflow as tf
import numpy as np
import pickle
from tensorflow import keras
from sklearn.model_selection import StratifiedKFold
from argparse import ArgumentParser
import matplotlib.pyplot as plt

########################
# Function Definitions #
########################

"""
## Getting the Data
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


def get_symmetric_normalized(A):
    """
    :param A: Adjacency matrix
    :return: symmetric normalized matrix A_tilda
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


########################
# Class Definitions    #
########################

"""
Class definition of node convolution layer
"""


class NodeConv(keras.layers.Layer):
    def __init__(self, num_classes, n_output_nodes, activation="relu", **kwargs):
        """
        :param num_classes: number of target classes
        :param n_output_nodes: number of output nodes
        :param activation: activation function
        :param kwargs: NA
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
        # w_init = keras.initializers.random_uniform(seed=None)
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


if __name__ == "__main__":

    """
    Command liner parser initialization
    """
    parser = ArgumentParser(description="Loading dataset")
    parser.add_argument("-data", dest="data", required=True,
                        help="input file with graphs dataset", metavar="FILE",
                        type=lambda x: is_valid_file(parser, x))
    parser.add_argument("-epoch", dest="epoch", required=True,
                        help="number of epochs for training", metavar="INT",
                        type=int)
    parser.add_argument("-drop", dest="drop", required=True,
                        help="floating point number to indicate drop rate", metavar="FLOAT",
                        type=float)

    args = parser.parse_args()

    """
    Load dataset
    """
    epoch = args.epoch
    drop_rate = args.drop  # drop rate
    graphs_data = pickle.load(args.data)

    if "ENZYMES" in str(args.data):

        """
        For ENZYMES,the node features are concatenating node attribute with the one-hot vectors of the node labels
        """
        padded_labels_ENZ = utils.get_padded_node_labels(graphs_data)

        for ind in range(len(graphs_data)):
            G = graphs_data[ind]
            nodes = np.array(G.nodes)
            num = len(nodes)
            for i in range(num):
                vec = G.nodes[nodes[i]]['node_attributes']
                # Normalize the node attributes in the l2-norm
                l2 = np.sqrt(np.sum(np.power(vec, 2)))
                label = padded_labels_ENZ[ind][i]
                G.nodes[nodes[i]]['node_attributes'] = np.append(vec / l2, label)

    else:
        """
        For NCI1,the node features are the one-hot vectors of the node labels.
        So we add the node feature to each node in the graph for every graph in graphs_NCI1
        """

        padded_labels_NC = utils.get_padded_node_labels(graphs_data)

        for ind in range(len(graphs_data)):
            G = graphs_data[ind]
            nodes = np.array(G.nodes)
            num = len(nodes)
            for i in range(num):
                G.nodes[nodes[i]]['node_attributes'] = padded_labels_NC[ind][i]

    # Get tensor of padded_node_attribute
    graphs = utils.get_padded_node_attributes(graphs_data)

    # Get the label of every graph in dataset
    graphs_label = np.array([utils.get_graph_label(G) for G in graphs_data])

    # num_nodes = max(|V|) of all the graphs in dataset
    num_nodes = graphs.shape[1]

    # Dimension of node attributes
    num_features = graphs.shape[2]

    # Number of graph labels
    num_classes = max(graphs_label)

    """
    3-D tensor adj_matrices stores the adjacent matrix of every graph in dataset.
    For graphs with less nodes we embed the adjacent matrix to a matrix of size num_nodes*num_nodes
    In this way they have sizes.
    """
    # number of graphs*num_nodes*num_nodes
    adj_matrices = np.zeros(shape=(len(graphs), num_nodes, num_nodes))
    for i in range(len(graphs)):
        adj = utils.get_adjacency_matrix(graphs_data[i])
        num_of_nodes = nx.number_of_nodes(graphs_data[i])
        adj_matrices[i][:num_of_nodes, :num_of_nodes] = adj
        adj_matrices[i] = get_symmetric_normalized(adj_matrices[i])

    # Convert graphs_label for later training
    Y = np.reshape(np.array(graphs_label), (len(graphs), 1))

    """
    Start of graph classification code
    """
    Kf = StratifiedKFold(n_splits=10, shuffle=True)
    Kf.get_n_splits(graphs_label)
    accuracy_test = []
    accuracy_train = []

    plot_acc_train = []
    plot_acc_test = []

    for train_index, test_index in Kf.split(graphs, graphs_label):
        # Inputs
        Input_adj = tf.keras.Input(shape=(num_nodes, num_nodes,))
        Input_X = tf.keras.Input(shape=(num_nodes, num_features,))

        # Layers
        layer1 = NodeConv(num_classes, 64, "relu")([Input_adj, Input_X])
        layer2 = NodeConv(num_classes, 64, "relu")([Input_adj, layer1])
        layer3 = NodeConv(num_classes, 64, "relu")([Input_adj, layer2])
        layer4 = NodeConv(num_classes, 64, "relu")([Input_adj, layer3])
        layer5 = NodeConv(num_classes, 64, "relu")([Input_adj, layer4])
        pool = PoolingLayer()(layer5)
        layer7 = tf.keras.layers.Dense(64, 'relu')(pool)
        # Dropout layer
        layer7 = tf.keras.layers.Dropout(drop_rate)(layer7)
        layer8 = tf.keras.layers.Dense(num_classes + 1, 'softmax')(layer7)

        # Model definition
        model = keras.Model(inputs=[Input_adj, Input_X], outputs=layer8)

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                      loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Model layout
        # if not os.path.exists("./graph_class_model.png"):
        #     keras.utils.plot_model(model, "graph_class_model.png", show_shapes=True, show_layer_names=False, rankdir='LR', dpi=256)

        # Model Training
        train = model.fit([adj_matrices[train_index], graphs[train_index]], Y[train_index], epochs=epoch, batch_size=16)

        # Model evaluation
        eva = model.evaluate([adj_matrices[test_index], graphs[test_index]], Y[test_index])

        # Append accuracy values
        accuracy_train.append(train.history['accuracy'][-1])
        accuracy_test.append(eva[1])

        plot_acc_train.append(train.history['accuracy'])
        plot_acc_test.append(eva[1])


    # Display mean accuracy and standard deviation
    print("***Statistics on Train Data***")
    print(f"Mean accuracy:  {np.mean(accuracy_train):0.04f}")
    print(f"Standard deviation: {np.std(accuracy_train):0.04f}")

    print("***Statistics on Test Data***")
    print(f"Mean accuracy:  {np.mean(accuracy_test):0.04f}")
    print(f"Standard deviation: {np.std(accuracy_test):0.04f}")

    fig, (ax0, ax1) = plt.subplots(ncols=2)

    #
    ax0.plot(plot_acc_train[0], 'r', plot_acc_train[1], 'b', plot_acc_train[2], 'y', plot_acc_train[3], 'm',
             plot_acc_train[4], 'g', plot_acc_train[5], '#00FFFF', plot_acc_train[6], '#00FF00', plot_acc_train[7],
             '#800000', plot_acc_train[8], '#808080', plot_acc_train[9], '#808000')

    # ax0.plot(plot_acc_train[0], 'r', plot_acc_train[1], 'b', plot_acc_train[2], 'y')
    ax0.legend(('Iter1', 'Iter2', 'Iter3', 'Iter4', 'Iter5', 'Iter6', 'Iter7', 'Iter8', 'Iter9', 'Iter10'))
    ax0.set_xlabel("Epochs")
    ax0.set_ylabel("Training Accuracy")

    x = np.arange(0, 10, 1)
    y_mean = [np.mean(plot_acc_test)] * len(x)
    ax1.set_ylim(0.3, 0.8)
    ax1.plot(x, plot_acc_test, marker='o')
    ax1.plot(x, y_mean, linestyle='--')
    ax1.legend(('Eval', 'Mean Accuracy'))
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Evaluation Accuracy")
    fig.suptitle('Training and Evaluation accuracies on ENZYMES', fontsize=16)
    plt.show()
