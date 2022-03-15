import os
import sys
sys.path.insert(0, '..')  # appending parent directory

# import other libraries
import numpy as np
import networkx as nx
import pickle
from sklearn.model_selection import KFold
from sklearn import svm
import random as random
import time
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

# import local libraries
import data_utils as utils
import cwkl as cwkl

# load DD dataset
with open('datasets/datasets/DD/data.pkl', 'rb') as f:
    graphs_DD = pickle.load(f)

# array for accuracy measure
acc = []

def condition(x):
    """
    This is a boolean function which indicates whether a graph has more than 1000 nodes.
    :param x: networkx graph object
    :return: boolean value
    """
    return len(x.nodes()) > 1000


# Remove graphs that have large nodes to ease computation
outliers = [idx for idx, element in enumerate(graphs_DD) if condition(element)]
for index in sorted(outliers, reverse=True):
    del graphs_DD[index]

# Shuffle the graphs
random.shuffle(graphs_DD)
walk_length = 5

# Time wrapper
tic = time.perf_counter()

# Get feature vectors for the dataset
feature_vectors = [cwkl.closed_walk(graph, walk_length) for graph in graphs_DD]

# computes how long feature vector generation took
toc = time.perf_counter()
print(f"Getting feature vector took {toc - tic:0.04f} seconds")

# Fit the feature vectors to min-max
Scaler = MinMaxScaler()
Scaler.fit(feature_vectors)
feature_vectors_scaled = Scaler.transform(feature_vectors)
feature_vectors_scaled = np.array(feature_vectors_scaled)

# Grab the labels and create an array
graph_label = [utils.get_graph_label(graph) for graph in graphs_DD]
graph_label = np.array(graph_label)

# Split dataset into 10 folds
Kf_DD = KFold(n_splits=10, shuffle=True)
Kf_DD.get_n_splits(graphs_DD)

# Initialize svm with 'precomputed' kernel
clf_DD = svm.SVC(kernel='precomputed')

# Start the training/validation loop
for train_index, test_index in Kf_DD.split(graphs_DD):
    # print("TRAIN:", train_index, "TEST:", test_index)
    graphs_DD_train, graphs_DD_test = feature_vectors_scaled[train_index], feature_vectors_scaled[test_index]
    graphs_DD_train_label, graphs_DD_test_label = graph_label[train_index], graph_label[test_index]

    # print("Fitting the SVM")
    # compute gram matrix from the feature vectors
    kernel_train = np.dot(graphs_DD_train, graphs_DD_train.T)

    # fit the svm
    clf_DD.fit(kernel_train, graphs_DD_train_label)

    # on test data
    kernel_test = np.dot(graphs_DD_test, graphs_DD_train.T)
    label_pred = clf_DD.predict(kernel_test)
    acc.append(accuracy_score(graphs_DD_test_label, label_pred))

# Display mean accuracy and standard deviation
print(f"The mean accuracy on the test data {np.mean(acc)}")
print(f"The standard deviation is {np.std(acc)}")