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


"""
## Get number of classes
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

    # TODO - Make this solution sparse
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
# Get maximum attribute length
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
        #w_init = keras.initializers.random_uniform(seed=None)
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


if __name__ == "__main__":
    """
    Command liner parser initialization
    """
    parser = ArgumentParser(description="Loading datasets")
    parser.add_argument("-train", dest="train", required=True,
                        help="input file with graphs train dataset", metavar="FILE",
                        type=lambda x: is_valid_file(parser, x))
    parser.add_argument("-test", dest="test", required=True,
                        help="input file with graphs test dataset", metavar="FILE",
                        type=lambda x: is_valid_file(parser, x))
    args = parser.parse_args()

    """
    Load Dataset
    """
    graph_train = pickle.load(args.train)
    graph_test = pickle.load(args.test)


    """
    Preprocess Data
    """
    # number of classes in data
    classes = get_node_class_list(graph_train)
    num_classes = len(classes)

    # number of features
    num_features = get_max_attr_length(graph_train)

    # number of nodes
    num_nodes = get_num_nodes(graph_train)

    # adjacency matrix
    graph_train = nx.Graph(graph_train[0])
    A = utils.get_adjacency_matrix(graph_train)
    A_tilda = get_symmetric_normalized(A)

    ## Train values
    X_train = utils.get_node_attributes(graph_train)
    Scaler = MinMaxScaler()
    Scaler.fit(X_train)
    X_train = Scaler.transform(X_train)
    X_train = np.array(X_train)
    Y_train = utils.get_node_labels(graph_train)

    ## Test values
    graph_test = nx.Graph(graph_test[0])
    X_test = utils.get_node_attributes(graph_test)
    Scaler = MinMaxScaler()
    Scaler.fit(X_test)
    X_test = Scaler.transform(X_test)
    X_test = np.array(X_test)
    Y_test = utils.get_node_labels(graph_test)

    accuracy_train = []
    accuracy_test = []

    plot_acc_train = []
    plot_acc_test = []

    for _ in range(10):
        # Model definition and initialization
        Input_adj = keras.Input(shape=(num_nodes, num_nodes,))
        Input_X = keras.Input(shape=(num_nodes, num_features,))
        hidden = NodeConv(n_output_nodes=32, num_classes=num_classes, activation="relu")([Input_adj, Input_X])
        hidden_2 = NodeConv(n_output_nodes=num_classes + 1, num_classes=num_classes, activation="softmax")(
            [Input_adj, hidden])
        model = keras.Model(inputs=[Input_adj, Input_X], outputs=hidden_2)
        model.compile(
            loss=keras.losses.sparse_categorical_crossentropy,
            optimizer="adam",
            metrics=[keras.metrics.sparse_categorical_accuracy],
        )

        # Model layout
        if not os.path.exists("./nodes_conv_model.png"):
            keras.utils.plot_model(model, "nodes_conv_model.png", show_shapes=True, dpi=256)

        # Reshape Inputs
        A_tilda_reshaped = np.reshape(A_tilda, (1, num_nodes, num_nodes))
        X_train_reshaped = np.reshape(X_train, (1, num_nodes, num_features))
        Y_train_reshaped = np.reshape(Y_train, (1, num_nodes))

        # Train
        train = model.fit(([A_tilda_reshaped, X_train_reshaped]), Y_train_reshaped - 1, epochs=80,
                  batch_size=len(Y_train_reshaped))

        # Evaluate
        print("\n\nAccuracy on Test Set")
        A_test = utils.get_adjacency_matrix(graph_test)
        A_tilda_test = get_symmetric_normalized(A_test)

        # Reshape Inputs
        A_tilda_test_reshaped = np.reshape(A_tilda_test, (1, num_nodes, num_nodes))
        X_test_reshaped = np.reshape(X_test, (1, num_nodes, num_features))
        Y_test_reshaped = np.reshape(Y_test, (1, num_nodes))

        # Evaluation on test dataset
        eva = model.evaluate([A_tilda_test_reshaped, X_test_reshaped], Y_test_reshaped - 1,
                             batch_size=len(Y_test_reshaped))

        # Append accuracy values
        accuracy_train.append(train.history['sparse_categorical_accuracy'][-1])
        accuracy_test.append(eva[1])

        plot_acc_train.append(train.history['sparse_categorical_accuracy'])
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

    #ax0.plot(plot_acc_train[0], 'r', plot_acc_train[1], 'b', plot_acc_train[2], 'y')
    ax0.legend(('Iter1', 'Iter2', 'Iter3', 'Iter4', 'Iter5', 'Iter6', 'Iter7', 'Iter8', 'Iter9', 'Iter10'))
    ax0.set_xlabel("Epochs")
    ax0.set_ylabel("Training Accuracy")

    x = np.arange(0,10,1)
    y_mean = [np.mean(plot_acc_test)]*len(x)
    ax1.set_ylim(0.60, 1)
    ax1.plot(x, plot_acc_test, marker='o')
    ax1.plot(x, y_mean, linestyle='--')
    ax1.legend(('Eval', 'Mean Accuracy'))
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Evaluation Accuracy")
    fig.suptitle('Training and Evaluation accuracies on Cora', fontsize=16)
    plt.show()