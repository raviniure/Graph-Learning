#!/usr/bin/env python
# coding: utf-8

# In[4]:


#Use graphlet kernel to train Support Vector Machines on three datasets DD
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


# In[5]:


#Function graph_atlas_g() return the list of all graphs with up to seven nodes.
g7=nx.graph_atlas_g()

#Extract all the five_node_graphlets from g7
five_node_graphlet = []
for graph in g7:
    if nx.number_of_nodes(graph) == 5:
        five_node_graphlet.append(graph)


# In[6]:


"""
Dataset DD
Form a dataframe for the feature vectors for later training. The number of rows is the lenth of dataset DD.
Every row corresponds a feature vector of a graph. The last column is the label of the graph
"""
with open('datasets/datasets/DD/data.pkl', 'rb') as f:
    graphs_DD = pickle.load(f)

#Initialize the dataframe
cols = np.arange(1,35,1)
cols = [str(n) for n in cols]
df = pd.DataFrame(columns = cols)

#Apply generate_feature_vec to every graph in datasets DD
###This step takes 10 minutes
for i in range(len(graphs_DD)):
    df.loc[i] = generate_feature_vec (graphs_DD[i],five_node_graphlet)

#Add column 'label' to the dataframe
graphs_DD_label = [utils.get_graph_label(graph) for graph in graphs_DD]
graphs_DD_label = np.array(graphs_DD_label)
df['label'] = graphs_DD_label

#The process of generating feature vectors takes relatively long. For convenience the dataframe can be stored in a csv file when needed
#df.to_csv (r'graphs_DD.csv', index = False, header=True)
#df = pd.read_csv("graphs_DD.csv")


# In[11]:


Kf_DD = KFold(n_splits=10,shuffle = True)
Kf_DD.get_n_splits(df[cols])
clf_DD = svm.SVC(kernel='precomputed',C=0.1)
acc=[]

for train_index, test_index in Kf_DD.split(df[cols]):
    graphs_DD_train, graphs_DD_test = df[cols].iloc[train_index], df[cols].iloc[test_index]
    graphs_DD_train_label, graphs_DD_test_label = df["label"].iloc[train_index], df["label"].iloc[test_index]
    
    #Train the model
    kernel_train = np.dot(graphs_DD_train,graphs_DD_train.T)
    clf_DD.fit(kernel_train,graphs_DD_train_label)
    
    #Predict on test data
    kernel_test = np.dot(graphs_DD_test,graphs_DD_train.T)
    label_pred = clf_DD.predict(kernel_test)
    
    #Calculte accuracy
    acc.append(accuracy_score(graphs_DD_test_label,label_pred))


# In[12]:


print("The ten accuracies are as follow")
print(acc)
print("")
print(f"The mean accuracy on the test data {np.mean(acc)}")
print(f"The standard deviation is {np.std(acc)}")


# In[ ]:




