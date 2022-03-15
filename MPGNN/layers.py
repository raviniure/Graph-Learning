#!/usr/bin/env python
# coding: utf-8
import tensorflow as tf
from tensorflow import keras


class Linear(keras.layers.Layer):
    """ Basic perceptron layer without bias"""

    def __init__(self, units=32):
        """
        :param units: the number of output units (default: 32)
        """
        super(Linear, self).__init__()
        self.units = units

    def build(self, input_shape):
        """
        :param input_shape: input tensor shape
        :return: NA
        """
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer="random_normal", trainable=True, )

    def call(self, inputs):
        """
        :param inputs: input tensor shape
        :return: matrix multiplication of inputs and W
        """
        return tf.matmul(inputs, self.w)


class MPGNN(keras.layers.Layer):
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
        if not self.res:
            return X
        else:
            if node_emd.shape[-1] == self.units:
                return node_emd + X
            else:
                return tf.matmul(node_emd, self.W) + X


class PoolingLayer(tf.keras.layers.Layer):
    """Pooling Layer"""

    def __init__(self, name="PoolingLayer", **kwargs):
        """
        :param name: "Pooling Layer"
        :param kwargs: NA
        """
        super(PoolingLayer, self).__init__(name=name, **kwargs)

    def call(self, inputs):
        """
        :param inputs: input tensor shape
        :return: Sum Pooling layer 
        """
        x = inputs
        pool = tf.reduce_sum(x, axis=1)
        return pool
