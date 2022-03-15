import os
import pickle

from argparse import ArgumentParser
import numpy as np
import pandas as pd
import data_utils as utils
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn import svm

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


def get_node_colors(G):
    """
    :param G: A networkx graph G=(V,E)
    :return: A numpy array of shape (|V|, a), where a is the length of the node colors vector
    """
    colors = np.int32([node[1]["color"] for node in G.nodes(data=True)])
    return colors


def initialize(G):
    """
    Function that initializes the node color to node label
    :param G: single networkx graph object
    :return: NA
    """
    for node in G.nodes:
        G.nodes[node]['color'] = G.nodes[node]['node_label']


def get_iteration_colors(G):
    """
    Function that computes a multiset of colors for each node in a graph G
    :param G: single networkx graph object
    :return:
    """
    nodes = np.array(G.nodes)
    color_list = []

    for node in nodes:
        # find all the neighbors for each node
        neighbors = G.neighbors(node)

        # get the color label for each neighbor of a node
        neighbors_label = [G.nodes[neighbor]['color'] for neighbor in neighbors]

        # sort the color labels
        new_color = np.sort(neighbors_label)

        # join the color labels and form a string
        new_color = ''.join(str(x) for x in new_color)

        # append the color list with the string formed from colors of neighbors
        color_list.append(new_color)

    return color_list


def update_iteration_colors(G, nodes, iter_strings, color_map):
    """
    Function that updates the multiset-color label to actual color according to color map.
    :param G: Networkx graph object
    :param nodes: nodes of current graph
    :param iter_strings: multiset of colors for the current graph, G
    :param color_map: multiset of color and the actual color label
    :return: NA
    """
    index = 0
    for node in nodes:
        G.nodes[node]['color'] = color_map[iter_strings[index]]
        index += 1


def get_count_labels(graph, node_labels):
    """
    Function that counts the number of occurences for each label class
    :param graph: networkx graph object
    :param node_labels: node labels
    :return: dictionary that maps label and their corresponding occurences in the graph
    """
    node_labels_graph = list(utils.get_node_labels(graph))
    count = dict((x, node_labels_graph.count(x)) for x in set(node_labels))
    return count


def check_stable(G):
    """
    Function that checks whether the graph G is stable in terms of color.
    :param G: networkx graph object
    :return: boolean to indicate whether the graph is stable or not, T if it is.
    """
    colors = len(set(get_node_colors(G)))

    # If all nodes are uniquely colored
    if colors == len(G.nodes):
        return True

    # Two nodes in same neighbourhood are colored same
    nodes = np.array(G.nodes)
    num_nodes = len(nodes)
    for i in range(num_nodes):
        for j in range(i, num_nodes):
            ind1 = [G.nodes[n]['color'] for n in G.neighbors(nodes[i])]
            ind2 = [G.nodes[n]['color'] for n in G.neighbors(nodes[j])]
            ind1 = np.sort(np.array(ind1))
            ind2 = np.sort(np.array(ind2))
            TF = (ind1 == ind2).all()
            if TF:
                return False
            else:
                return True


def get_feature_vector(graphs, iter):
    """
    Function that obtains the WL features
    :param graphs: list of graphs
    :param iter: number of rounds of color refinement
    :return: dataframe with colors as the column name and rows as a particular graph.
            entries contain the number of occurence of each color in the column for the graph.
    """
    # Get all possible node labels in the graphs dataset
    # Some graphs may not contain a label but that is okay.
    all_node_labels = set()
    for graph in graphs:
        # get all the different labels
        all_node_labels.update(utils.get_node_labels(graph))
        # also initialize the graphs with their node label as color label
        initialize(graph)

    # define a dataframe that stores histogram of color labels for each iteration
    # initially the dataframe only consists of column as all node labels
    # there will be new columns with every iterations
    feature_vector_df = pd.DataFrame(columns=list(all_node_labels))

    # update the feature vector for each graph with counts of node labels
    # update row for each graph
    for index in range(len(graphs)):
        count_labels = get_count_labels(graphs[index], all_node_labels)
        feature_vector_df.loc[index] = count_labels

    # We will need some form of global string-label to label map
    map_labels = {}
    for label in all_node_labels:
        map_labels.update({str(label): label})

    # begin the iterations
    for itr in range(iter):
        for graph in graphs:
            # if graph nodes are unique, move on
            if check_stable(graph):
                break

            # get the strings of multiset of colors for this graph from this iteration
            itr_color_strings = get_iteration_colors(graph)

            # find newer node labels generated for graph during this iteration
            new_node_labels = set(itr_color_strings)
            new_node_labels = np.sort(list(new_node_labels))

            # map each new node labels to new color if it doesn't exists in global map
            for new_label in new_node_labels:
                if str(new_label) not in map_labels.keys():
                    new_color = max(all_node_labels) + 1
                    map_labels.update({str(new_label): new_color})
                    all_node_labels.update([new_color])

            # update graph with the newly obtained color labels
            update_iteration_colors(graph, graph.nodes, itr_color_strings, map_labels)

        # An array contains the number of colors of each graph
        number_of_color_r = [max(get_node_colors(G)) for G in graphs]
        # The dimension of the feature vectors in round r
        dim_r = max(number_of_color_r)
        vec_dim_r = np.arange(1, dim_r + 1, 1)
        vec_dim_r = [str(n) for n in vec_dim_r]
        vec_df_r = pd.DataFrame(columns=vec_dim_r)

        for ind in range(len(graphs)):
            cols_r = get_node_colors(graphs[ind])
            count_r = np.zeros(dim_r)
            for col in cols_r:
                count_r[col - 1] = count_r[col - 1] + 1
            vec_df_r.loc[ind] = count_r

        # Concatenate the original dataframe with the new one
        # If there are existing nodes with color that didn't change compared to last iterations,
        # the new vec_df_r would replace it.
        feature_vector_df = pd.concat([feature_vector_df, vec_df_r], axis=1, ignore_index=True)

    return feature_vector_df


def compute_dot(a, b):
    """
    Function to compute dot product of two matrices
    :param a: matrix a
    :param b: matrix b
    :return: dot product
    """
    return np.dot(a, b)


def train(data_train, data_eval):

    combined = [*data_train, *data_eval]

    """
    Obtain graph labels for train dataset
    """
    graph_label = [utils.get_graph_label(graph) for graph in data_train]
    graph_label = np.array(graph_label)

    """
    Obtain feature vectors for the combined dataset
    """
    feature_df = get_feature_vector(combined, 4)

    """
    Separate out training feature set and evaluation feature set
    """
    train_feature = feature_df.iloc[:len(data_train), :]
    eval_feature = feature_df.iloc[len(data_train):, :]

    """
    Obtain all the different colors present in the training features
    """
    colors = list(train_feature)

    """
    Define Kfolds
    """
    Kf = KFold(n_splits=10, shuffle=True)
    Kf.get_n_splits(train_feature[colors])
    # Kf = StratifiedKFold(n_splits=10, shuffle=True)
    # Kf.get_n_splits(train_feature[colors], graph_label)

    """
    Initialize SVM classifier
    """
    clf = svm.SVC(kernel='precomputed', C=100)

    """
    Empty accuracy list
    """
    acc = []

    """
    Perform K-fold cross validation
    """
    for train_index, valid_index in Kf.split(train_feature[colors]):
        graphs_train, graphs_valid = train_feature[colors].iloc[train_index], train_feature[colors].iloc[valid_index]
        graphs_train_label, graphs_valid_label = graph_label[train_index], graph_label[valid_index]

        # Convert df to numpy to make training quicker
        graphs_train = graphs_train.to_numpy(dtype=np.float32)
        graphs_valid = graphs_valid.to_numpy(dtype=np.float32)

        # Obtain gram matrix
        kernel_train = np.dot(graphs_train, graphs_train.T)

        # train the classifier
        clf.fit(kernel_train, graphs_train_label)

        # Obtain gram matrix for validation
        kernel_valid = np.dot(graphs_valid, graphs_train.T)

        # Obtain predictions
        label_pred = clf.predict(kernel_valid)

        # Calculate accuracy and append to accuracy list
        acc.append(accuracy_score(graphs_valid_label, label_pred))

    print("Training Accuracies:")
    print(acc)
    print("")
    print(f"Mean Training Accuracy: {np.mean(acc):0.04f}")
    print(f"Standard Deviation: {np.std(acc):0.04f}")

    # Evaluation set
    graph_label_eval = [utils.get_graph_label(graph) for graph in data_eval]
    graph_label_eval = np.array(graph_label_eval)

    eval_feature = eval_feature[colors].iloc[:]
    eval_feature = eval_feature.to_numpy(dtype=np.float32)

    # print(np.shape(eval_feature), np.shape(graphs_train.T))
    kernel_eval = np.dot(eval_feature, graphs_train.T)
    label_pred_eval = clf.predict(kernel_eval)

    acc_eval = accuracy_score(graph_label_eval, label_pred_eval)

    print("----------------------------------------")
    print(f"Evaluation accuracy: {acc_eval:0.04f}")
