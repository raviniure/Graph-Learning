import networkx as nx
import tensorflow as tf
import numpy as np
import pickle
from tensorflow import keras
from argparse import ArgumentParser
import random

from data_utils import get_node_labels, get_graph_label

from functions import get_edge_label, incidence_generate, get_maximums, get_padded
from layers import Linear, MPGNN, PoolingLayer

if __name__ == "__main__":

    """
    Command liner parser initialization
    """
    parser = ArgumentParser(description="Hyperparameters of the network")

    parser.add_argument("-weighted", dest="weighted", required=True,
                        help="Incidence matrices are weighted or not", metavar="INT", type=int)

    parser.add_argument("-num", dest="num", required=True,
                        help="Number of MPGNN layers", metavar="INT", type=int)

    parser.add_argument("-rc", dest="rc", required=True,
                        help="Whether with residual connected or not", metavar="INT", type=int)

    parser.add_argument("-units", dest="units", required=True,
                        help="Number of output units of MPGNN layers", metavar="INT", type=int)

    parser.add_argument("-epoch", dest="epoch", required=True,
                        help="Number of epochs for training", metavar="INT", type=int)

    args = parser.parse_args()

    weighted = args.weighted
    num = args.num
    rc = args.rc
    units = args.units
    epoch = args.epoch

    """Import data for training, validation and test"""

    with open('datasets/Zinc_Train/data.pkl', 'rb') as f:
        data_train = pickle.load(f)  # length: 10000

    with open('datasets/Zinc_Val/data.pkl', 'rb') as f:
        data_val = pickle.load(f)  # length: 1000

    with open('datasets/Zinc_Test/data.pkl', 'rb') as f:
        data_test = pickle.load(f)  # length: 1000

    # Get Graph labels for training, validation and evaluation sets
    train_label = [get_graph_label(G) for G in data_train]
    train_label = np.reshape(np.array(train_label), (len(train_label), 1))

    val_label = [get_graph_label(G) for G in data_val]
    val_label = np.reshape(np.array(val_label), (len(val_label), 1))

    test_label = [get_graph_label(G) for G in data_test]
    test_label = np.reshape(np.array(test_label), (len(test_label), 1))

    # Get the max number of edges & nodes as well as number of edge & node labels across whole Zinc dataset
    # in order to get the uniform size of training, validation and test data
    """
    max number of edges: 42,
    max number of nodes: 37,
    edge labels:      [1, 2, 3],
    node labels:      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    """
    max_num_edge, max_num_node, edge_labels, node_labels = get_maximums(data_train, data_val, data_test)

    # Get padded input of train, validation and test datasets
    input_train = get_padded(data_train, max_num_node, max_num_edge, edge_labels, node_labels, weighted=weighted)
    input_val = get_padded(data_val, max_num_node, max_num_edge, edge_labels, node_labels, weighted=weighted)
    input_test = get_padded(data_test, max_num_node, max_num_edge, edge_labels, node_labels, weighted=weighted)
    """
    input:       
    node embedding X: (batch_size,|V|,    node_label_numbers)
    edge feature F:   (batch_size,|E|*2， edge_label_numbers=dim_k^E)
    I_out:            (batch_size,|E|*2,  |V|)
    I_in:             (batch_size,|E|*2,  |V|)
    """

    # Define the input of the model
    input_node = tf.keras.Input(shape=(max_num_node, len(node_labels),))
    input_edge = tf.keras.Input(shape=(2 * max_num_edge, len(edge_labels),))
    input_iout = tf.keras.Input(shape=(2 * max_num_edge, max_num_node,))
    input_iin = tf.keras.Input(shape=(2 * max_num_edge, max_num_node,))

    """Build the model"""

    # First mpgnn layer which will be different than the rest as input will be input_node
    mpgnn = MPGNN(units=units, res=rc)([input_node, input_edge, input_iout, input_iin])

    # second and later mpgnn layers with output of the first mpgnn as input
    for i in range(num - 1):
        mpgnn = MPGNN(units=units, res=rc)([mpgnn, input_edge, input_iout, input_iin])

    # pooling layer after mpgnns
    pool = PoolingLayer()(mpgnn)

    # dense layers
    dense1 = tf.keras.layers.Dense(units, 'relu')(pool)
    dense = tf.keras.layers.Dense(1)(dense1)

    # model initialization
    model = keras.Model(inputs=[input_node, input_edge, input_iout, input_iin], outputs=dense)

    # compile model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2 * 1e-3),
                  loss=tf.keras.losses.MeanAbsoluteError())

    train_model = model.fit(input_train, train_label, epochs=epoch, batch_size=32)

    print("***Mean Absolute Error on Train Data***")
    train_eval = model.evaluate(input_train, train_label)
    
    print("***Mean Absolute Error on Validation Data***")
    val = model.evaluate(input_val, val_label)

    print("***Mean Absolute Error on Test Data***")
    test = model.evaluate(input_test, test_label)