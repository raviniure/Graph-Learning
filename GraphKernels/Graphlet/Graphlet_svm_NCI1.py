#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Use graphlet kernel to train Support Vector Machines on three datasets NCI1
import numpy as np
import networkx as nx
import random
import pickle
import pandas as pd

from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.metrics import accuracy_score
import data_utils as utils
from Graphlet import generate_feature_vec

g7=nx.graph_atlas_g()
five_node_graphlet = []
for graph in g7:
    if nx.number_of_nodes(graph) == 5:
        five_node_graphlet.append(graph)


# In[2]:


#############################
#####Dataset graphs_NCI1#####
#############################
with open('datasets/datasets/NCI1/data.pkl', 'rb') as f:
    graphs_NCI1 = pickle.load(f)

col = np.arange(1,35,1)
col = [str(n) for n in col]
df3 = pd.DataFrame(columns = col)

#It takes around 36 minutes
for i in range(len(graphs_NCI1)):
    if graphs_NCI1[i].number_of_nodes()<5:
        df3.loc[i] = np.zeros(34)
    else:
        df3.loc[i] = generate_feature_vec(graphs_NCI1[i],five_node_graphlet)

graphs_NCI1_label = [utils.get_graph_label(graph) for graph in graphs_NCI1]
graphs_NCI1_label = np.array(graphs_NCI1_label)
df3['label'] = graphs_NCI1_label

#df3.to_csv (r'graphs_NCI1.csv', index = False, header=True)
#df3 = pd.read_csv("graph_NCI1.csv")


# In[8]:


Kf_NC = KFold(n_splits=10,shuffle = True)
Kf_NC.get_n_splits(df3[col])
clf_NC = svm.SVC(kernel='precomputed',C=0.01)
acc3=[]

for train_index, test_index in Kf_NC.split(df3[col]):
    graphs_NCI1_train, graphs_NCI1_test = df3[col].iloc[train_index], df3[col].iloc[test_index]
    graphs_NCI1_train_label, graphs_NCI1_test_label = df3["label"].iloc[train_index], df3["label"].iloc[test_index]
    
    kernel_train3 = np.dot(graphs_NCI1_train,graphs_NCI1_train.T)
    clf_NC.fit(kernel_train3,graphs_NCI1_train_label)
    
    kernel_test3 = np.dot(graphs_NCI1_test,graphs_NCI1_train.T)
    label_pred3 = clf_NC.predict(kernel_test3)
    acc3.append(accuracy_score(graphs_NCI1_test_label,label_pred3))


# In[11]:


print("The ten accuracies are as follow")
print(acc3)
print("")
print(f"The mean accuracy on the test data {np.mean(acc3)}")
print(f"The standard deviation is {np.std(acc3)}")


# In[ ]:




