import os
import sys
sys.path.insert(0, '..');  # appending parent directory

import numpy as np
import scipy as sp
import data_utils as utils

def closed_walk(G, len):
    """
    This function computes closed walks of length |len| on graph 'G'.
    :param
        - 'G' - networkx.Graph object
        - 'len' - length of walk
    :return
        - feature vector of dimension |len| that estimates the closed walks of length |len|
    """

    # adjacency matrix which encodes the connection between nodes
    adjacency = utils.get_adjacency_matrix(G)
    feature_vector = np.zeros(len)

    # Adjacency matrix when powered to k gives walks of k length
    # The diagonal entries of the powered adjacency matrix give the closed walks.
    for walk_length in range(len):
        feature_vector[walk_length] = np.trace(np.linalg.matrix_power(adjacency, walk_length+1))

    # return feature vector
    return feature_vector