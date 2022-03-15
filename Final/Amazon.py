#!/usr/bin/env python
# coding: utf-8
import networkx as nx
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
from tensorflow import keras
from argparse import ArgumentParser
import itertools

from sklearn.metrics import accuracy_score, roc_auc_score

from data_utils import get_adjacency_matrix, get_node_labels, get_graph_label
from data_utils import get_padded_node_labels, get_node_attributes, get_padded_node_attributes

# GCN
from gcn import is_valid_file, get_node_class_list, get_symmetric_normalized, get_max_attr_length, NodeConv

"""
Amazon is an 8-class node classification dataset. 
Your task is to predict the node label based on 
the Node Attributes and the Graph Structure.
"""


def train(az_train, az_val):
    graph_train = az_train[0]
    graph_val = az_val[0]

    nn_train = nx.number_of_nodes(graph_train)  # number of nodes of training graph
    nn_val = nx.number_of_nodes(graph_val)  # number of nodes of evaluation graph

    # Pad Node Attributes
    pad = get_padded_node_attributes([graph_train, graph_val])  # [2, 2550, 750]

    # Max number of nodes
    nn = pad[0].shape[0]
    classes = get_node_class_list(az_train)  # {0, 1, 2, 3, 4, 5, 6, 7}

    # Number of Node Classes
    num_classes = len(classes)

    # Dimension of Feature
    num_features = pad[0].shape[1]  # 745

    X_train = pad[0]
    X_val = pad[1]

    # Adjacency matrix
    A_train = get_adjacency_matrix(graph_train)
    A_val = get_adjacency_matrix(graph_val)

    A_tilda_train = get_symmetric_normalized(A_train)
    A_tilda_val = get_symmetric_normalized(A_val)

    # Node labels
    Y_train = get_node_labels(graph_train)
    Y_val = get_node_labels(graph_val)

    # Pad the adjacent matrices 
    if nn_train < nn_val:
        for i in range(nn_val - nn_train):
            Y_train = np.append(Y_train, 0)
        C = np.zeros(shape=(nn_val, nn_val))
        C[:nn_train, :nn_train] = A_tilda_train
        A_tilda_train = C
    else:
        for i in range(nn_train - nn_val):
            Y_val = np.append(Y_val, 0)
        C = np.zeros(shape=(nn_train, nn_train))
        C[:nn_val, :nn_val] = A_tilda_val
        A_tilda_val = C

    Input_adj = keras.Input(shape=(nn, nn,))
    Input_X = keras.Input(shape=(nn, num_features,))

    """Build the Model:2-layer GCN
    """
    hidden = NodeConv(n_output_nodes=160, num_classes=num_classes, activation="relu")([Input_adj, Input_X])
    hidden_2 = NodeConv(n_output_nodes=num_classes, num_classes=num_classes, activation="softmax")([Input_adj, hidden])
    model = keras.Model(inputs=[Input_adj, Input_X], outputs=hidden_2)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics=[keras.metrics.sparse_categorical_accuracy], )

    # Reshape Inputs
    A_tilda_reshaped = np.reshape(A_tilda_train, (1, nn, nn))
    X_train_reshaped = np.reshape(X_train, (1, nn, num_features))
    Y_train_reshaped = np.reshape(Y_train, (1, nn))

    # Training
    train = model.fit(([A_tilda_reshaped, X_train_reshaped]), Y_train_reshaped, epochs=120,
                      batch_size=len(Y_train_reshaped))

    # Reshape inputs for validation
    A_tilda_val_reshaped = np.reshape(A_tilda_val, (1, nn, nn))
    X_val_reshaped = np.reshape(X_val, (1, nn, num_features))
    Y_val_reshaped = np.reshape(Y_val, (1, nn))

    print("Accuracy on Training Set")
    print(model.evaluate([A_tilda_reshaped, X_train_reshaped], Y_train_reshaped, batch_size=1)[1])
    # Evaluation on test dataset
    print("Accuracy on Validation Set")
    print(model.evaluate([A_tilda_val_reshaped, X_val_reshaped], Y_val_reshaped, batch_size=1)[1])
