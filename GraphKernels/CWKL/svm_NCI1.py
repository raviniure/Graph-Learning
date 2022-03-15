import os
import sys
sys.path.insert(0,'..') # appending parent directory

# import external libraries
import numpy as np
import networkx as nx
import pickle
from sklearn.model_selection import KFold
from sklearn import svm
import random as random
import time
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

# import local libraries
import data_utils as utils
import cwkl as cwkl

# load NCI1 dataset
with open('datasets/datasets/NCI1/data.pkl', 'rb') as f:
    graphs_NCI1 = pickle.load(f)

# array for accuracy measure
acc = []

# Shuffle the graphs
random.shuffle(graphs_NCI1)
walk_length = 5

# Time wrapper
tic = time.perf_counter()

# Get feature vectors for the dataset
feature_vectors = [cwkl.closed_walk(graph, walk_length) for graph in graphs_NCI1]

# Computes how long feature vector generation took
toc = time.perf_counter()
print(f"Getting feature vector took {toc - tic:0.04f} seconds")

# Fit the feature vectors to min-max
Scaler = MinMaxScaler()
Scaler.fit(feature_vectors)
feature_vectors_scaled = Scaler.transform(feature_vectors)
feature_vectors_scaled = np.array(feature_vectors_scaled)

# Grab the labels and create an array
graph_label = [utils.get_graph_label(graph) for graph in graphs_NCI1]
graph_label = np.array(graph_label)

# Split dataset into 10 folds
Kf_NCI1 = KFold(n_splits=10, shuffle=True)
Kf_NCI1.get_n_splits(graphs_NCI1)

# Initialize svm with 'precomputed' kernel and 'C' as 100
clf_NCI1 = svm.SVC(kernel='precomputed', C = 100)

# Start the training/validation loop
for train_index, test_index in Kf_NCI1.split(graphs_NCI1):
    # print("TRAIN:", train_index, "TEST:", test_index)
    graphs_NCI1_train, graphs_NCI1_test = feature_vectors_scaled[train_index], feature_vectors_scaled[test_index]
    graphs_NCI1_train_label, graphs_NCI1_test_label = graph_label[train_index], graph_label[test_index]

    # print("Fitting the SVM")
    # compute gram matrix from the feature vectors
    kernel_train = np.dot(graphs_NCI1_train, graphs_NCI1_train.T)

    # fit the svm
    clf_NCI1.fit(kernel_train, graphs_NCI1_train_label)

    # on test data
    kernel_test = np.dot(graphs_NCI1_test, graphs_NCI1_train.T)
    label_pred = clf_NCI1.predict(kernel_test)
    acc.append(accuracy_score(graphs_NCI1_test_label, label_pred))

# Display mean accuracy and standard deviation
print(f"The mean accuracy on the test data {np.mean(acc)}")
print(f"The standard deviation is {np.std(acc)}")