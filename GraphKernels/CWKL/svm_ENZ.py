import os
import sys

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0,'..') # appending parent directory

# import other libraries
import numpy as np
import networkx as nx
import pickle
from sklearn.model_selection import KFold
from sklearn import svm
import random as random
import time
import pandas as pd

# import local libraries
import data_utils as utils
import cwkl as cwkl

# load ENZ dataset
with open('datasets/datasets/ENZYMES/data.pkl', 'rb') as f:
    graphs_ENZ = pickle.load(f)

# array for accuracy measure
acc = []

# Shuffle the graphs
random.shuffle(graphs_ENZ)
walk_length = 5

# Time wrapper
tic = time.perf_counter()

# Get feature vectors for the dataset
feature_vectors = [cwkl.closed_walk(graph, walk_length) for graph in graphs_ENZ]

# computes how long feature vector generation took
toc = time.perf_counter()
print(f"Getting feature vector took {toc - tic:0.04f} seconds")

# Fit the feature vectors to min-max
Scaler = MinMaxScaler()
Scaler.fit(feature_vectors)
feature_vectors_scaled = Scaler.transform(feature_vectors)
feature_vectors_scaled = np.array(feature_vectors_scaled)

# Grab the labels and create an array
graph_label = [utils.get_graph_label(graph) for graph in graphs_ENZ]
graph_label = np.array(graph_label)

# Split the dataset into 10 folds
Kf_ENZ = KFold(n_splits=10)
Kf_ENZ.get_n_splits(graphs_ENZ)

# Initialize svm with 'precomputed' kernel
clf_ENZ = svm.SVC(kernel='precomputed')


for train_index, test_index in Kf_ENZ.split(graphs_ENZ):
    # print("TRAIN:", train_index, "TEST:", test_index)

    graphs_ENZ_train, graphs_ENZ_test = feature_vectors_scaled[train_index], feature_vectors_scaled[test_index]
    graphs_ENZ_train_label, graphs_ENZ_test_label = graph_label[train_index], graph_label[test_index]

    # print("Fitting the SVM")
    # compute gram matrix from the feature vectors
    kernel_train = np.dot(graphs_ENZ_train, graphs_ENZ_train.T)

    # fit the svm
    clf_ENZ.fit(kernel_train, graphs_ENZ_train_label)

    # on test data
    kernel_test = np.dot(graphs_ENZ_test, graphs_ENZ_train.T)
    label_pred = clf_ENZ.predict(kernel_test)
    acc.append(accuracy_score(graphs_ENZ_test_label, label_pred))

# Display mean accuracy and standard deviation
print(f"The mean accuracy on the test data {np.mean(acc)}")
print(f"The standard deviation is {np.std(acc)}")