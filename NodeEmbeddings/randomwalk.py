import os
import numpy as np
import networkx as nx
import random

########################
# Function Definitions #
########################

def is_valid_file(_parser, arg):
    """
    :param _parser: parser object that was initialized
    :param arg: io.TextIOWrapper
    :return: return an open file handle
    """
    if not os.path.exists(arg):
        _parser.error("The file %s does not exist!" % arg)
    else:
        return open(arg, 'rb')  # return an open file handle


def get_distance(_graph, previous, target):
    """
    Function that calculates the distance between previous and target node
    in the given graph
    :param _graph: networkx graph
    :param previous: node in graph
    :param target: node in graph
    :return: integer valued distance between previous and target
    """

    return nx.shortest_path_length(_graph, source=previous, target=target)


def p_randomwalk(_graph, _start, _p, _q, _steps):
    """
    Performs parametrized random walk and returns walk sequence in a graph.
    :param _graph: networkx graph
    :param _start: starting node
    :param _p: probability of going back or BFS parameter
    :param _q: probability of going deep or DFS parameter
    :param _steps: length of the random walk desired
    :return: walk sequence of length = |_steps|
    """

    walk_sequence = []

    curr_node = _start
    prev_node = _start

    for step in range(_steps):
        # obtain neighbors
        neighbors = list(nx.neighbors(_graph, curr_node))
        num_neighbors = len(neighbors)

        # bias array
        bias = [0] * num_neighbors

        # for each neighbor, calculate bias
        for index in range(num_neighbors):
            distance = get_distance(_graph, prev_node, neighbors[index])
            if distance == 0:
                bias[index] = 1 / _p
            elif distance == 1:
                bias[index] = 1
            else:
                bias[index] = 1 / _q

        # obtain distribution
        normalization = np.sum(bias)
        distribution = bias / normalization

        # debug point
        assert 1 - sum(distribution) <= 1e-7, "Probability doesn't sum to 1"

        # sample a node using above distribution
        target = np.random.choice(neighbors, replace=True, p=distribution)
        walk_sequence.append(target)

        # update current and previous
        prev_node = curr_node
        curr_node = target

    return walk_sequence


def get_negative_sample(_graph, positive):
    """
    Computes a list of negative samples given a list of positive sample
    :param _graph: networkx graph
    :param positive: list that contains networkx nodes
    :return: list of networkx nodes not in positive
    """

    _nodes = list(set(_graph.nodes) - set(positive))
    negative_sample = random.sample(_nodes, len(positive))
    return negative_sample
