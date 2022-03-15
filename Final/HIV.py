import tensorflow as tf
import numpy as np
from tensorflow import keras

from sklearn.metrics import accuracy_score, roc_auc_score

from data_utils import get_adjacency_matrix, get_node_labels, get_graph_label, get_node_attributes
from data_utils import get_padded_node_labels, get_node_attributes, get_padded_node_attributes

from functions_layers import get_edge_attributes, get_padded_edge_attributes, double, incidence_generate, get_maximums, \
    get_padded
from functions_layers import Linear, MPGNN, PoolingLayer


def train(hiv_train, hiv_val):
    # Get Graph labels for training and validation sets
    train_label = [int(get_graph_label(G)) for G in hiv_train]
    val_label = [int(get_graph_label(G)) for G in hiv_val]

    train_label = np.reshape(np.array(train_label), (len(train_label), 1))
    val_label = np.reshape(np.array(val_label), (len(val_label), 1))

    len_train = len(hiv_train)
    len_va = len(hiv_val)

    """Node Attributes"""
    node_features_train = get_padded_node_attributes(hiv_train + hiv_val)[:len_train]  # (9760, 74, 86)
    node_features_val = get_padded_node_attributes(hiv_train + hiv_val)[len_train:]

    """Edges Attribute"""
    edge_features_train = double(get_padded_edge_attributes(hiv_train + hiv_val)[:len_train])  # (9760, 164, 6)
    edge_features_val = double(get_padded_edge_attributes(hiv_train + hiv_val)[len_train:])

    input_train = list((node_features_train, edge_features_train))
    input_val = list((node_features_val, edge_features_val))

    max_num_edge, max_num_node = get_maximums(hiv_train, hiv_val)

    input_train = input_train + list(get_padded(hiv_train, max_num_edge, max_num_node))  # + I_out, I_in
    input_val = input_val + list(get_padded(hiv_val, max_num_edge, max_num_node))

    # Define the input of the model
    input_node = tf.keras.Input(shape=(max_num_node, node_features_train.shape[2],))
    input_edge = tf.keras.Input(shape=(2 * max_num_edge, edge_features_train.shape[2],))
    input_iout = tf.keras.Input(shape=(2 * max_num_edge, max_num_node,))
    input_iin = tf.keras.Input(shape=(2 * max_num_edge, max_num_node,))

    # Output units
    units = 64
    # First mpgnn layer
    mpgnn = MPGNN(units=units, res=True)([input_node, input_edge, input_iout, input_iin])
    # In total 4 mpgnn layers
    for i in range(3):
        mpgnn = MPGNN(units=units, res=True)([mpgnn, input_edge, input_iout, input_iin])

    # Pooling layer after MPGNN layers
    pool = PoolingLayer()(mpgnn)

    # Dense layers
    dense = tf.keras.layers.Dense(units, "relu")(pool)
    # Add a drop out layer to avoid overfitting 
    dense = tf.keras.layers.Dropout(0.2)(dense)
    dense = tf.keras.layers.Dense(1, "sigmoid")(dense)

    # model initialization
    model = keras.Model(inputs=[input_node, input_edge, input_iout, input_iin], outputs=dense)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2 * 1e-3),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=["accuracy"], )

    model.fit(input_train, train_label, epochs=40, batch_size=64)

    print("The roc_auc_score on training dataset is: ")
    print(roc_auc_score(train_label, model.predict(input_train)))
    print("The roc_auc_score on test dataset is: ")
    print(roc_auc_score(val_label, model.predict(input_val)))
