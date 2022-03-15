import networkx as nx
import tensorflow as tf
import numpy as np
import pickle
from tensorflow import keras
from argparse import ArgumentParser
from sklearn.metrics import accuracy_score
import itertools

from data_utils import get_adjacency_matrix, get_node_attributes
from data_utils import get_node_labels, get_graph_label
from data_utils import get_padded_adjacency, get_padded_node_labels, get_padded_node_attributes

from gcn import get_node_class_list, get_symmetric_normalized, get_max_attr_length
from gcn import is_valid_file, NodeConv, PoolingLayer


def train(data_train, data_val):
    # Pad node labels
    pad_label = get_padded_node_labels(data_train + data_val)

    # Node features for training and validation sets
    node_features_train = pad_label[:len(data_train)]
    node_features_val = pad_label[len(data_train):]

    # Get Graph labels for training and validation sets
    train_label = [get_graph_label(G) for G in data_train]
    train_label = np.reshape(np.array(train_label), (len(train_label), 1))

    val_label = [get_graph_label(G) for G in data_val]
    val_label = np.reshape(np.array(val_label), (len(val_label), 1))

    # Adjacent matrices for training and validation sets
    adj_matrices_train = get_padded_adjacency(data_train + data_val)[:len(data_train)]
    adj_matrices_val = get_padded_adjacency(data_train + data_val)[len(data_train):]

    for i in range(len(adj_matrices_train)):
        adj_matrices_train[i] = get_symmetric_normalized(adj_matrices_train[i])
    for i in range(len(adj_matrices_val)):
        adj_matrices_val[i] = get_symmetric_normalized(adj_matrices_val[i])

    max_num_node = adj_matrices_train.shape[1]
    node_labels = node_features_train.shape[2]

    # Hyper-parameters
    drop_rate = 0.15  # drop rate
    l_r = 5 * 1e-3  # learning rate
    units = 64  # output units
    epochs = 45  # 30
    batch_size = 128

    # Define the Inputs of the model
    Input_adj = tf.keras.Input(shape=(max_num_node, max_num_node,))
    Input_X = tf.keras.Input(shape=(max_num_node, node_labels,))

    # Layers
    layer1 = NodeConv(2, units, "relu")([Input_adj, Input_X])
    layer2 = NodeConv(2, units, "relu")([Input_adj, layer1])
    layer3 = NodeConv(2, units, "relu")([Input_adj, layer2])
    layer4 = NodeConv(2, units, "relu")([Input_adj, layer3])
    pool = PoolingLayer()(layer4)
    layer6 = tf.keras.layers.Dense(units, 'relu')(pool)
    # Dropout layer
    layer7 = tf.keras.layers.Dropout(drop_rate)(layer6)
    layer8 = tf.keras.layers.Dense(2, 'softmax')(layer7)

    # Model definition
    model = keras.Model(inputs=[Input_adj, Input_X], outputs=layer8)

    # Compile model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=l_r),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Model Training
    train = model.fit([adj_matrices_train, node_features_train], train_label, epochs=epochs, batch_size=batch_size)

    # Model evaluation
    evaluation = model.evaluate([adj_matrices_val, node_features_val], val_label)

    accuracy_train = []
    accuracy_test = []

    # Accuracy values
    accuracy_train.append(train.history['accuracy'][-1])
    accuracy_test.append(evaluation[1])

    print("***Statistics on Train Data***")
    print(f"Accuracy:  {np.mean(accuracy_train):0.04f}")

    print("***Statistics on Validation Data***")
    print(f"Accuracy:  {np.mean(accuracy_test):0.04f}")
