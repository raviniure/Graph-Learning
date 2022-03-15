#!/usr/bin/env python
# coding: utf-8

# In[30]:


#Use graphlet kernel to train Support Vector Machines on three datasets DD, ENZYMES, and NCI1
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


# In[31]:


#Function graph_atlas_g() return the list of all graphs with up to seven nodes.
g7=nx.graph_atlas_g()

#Extract all the five_node_graphlets from g7
five_node_graphlet = []
for graph in g7:
    if nx.number_of_nodes(graph) == 5:
        five_node_graphlet.append(graph)


# In[19]:


##########################
####Dataset graphs_ENZ####
##########################
with open('datasets/datasets/ENZYMES/data.pkl', 'rb') as f:
    graphs_ENZ = pickle.load(f)

col = np.arange(1,35,1)
col = [str(n) for n in col]
df2 = pd.DataFrame(columns = col)

#Apply generate_feature_vec to every element of ENZ
#This step takes around 7 minutes#
for i in range(len(graphs_ENZ)):
    if graphs_ENZ[i].number_of_nodes()<5:
        df2.loc[i] = np.zeros(34)
    else:
        df2.loc[i] = generate_feature_vec (graphs_ENZ[i],five_node_graphlet)

#Add column 'label' to the dataframe
graphs_ENZ_label = [utils.get_graph_label(graph) for graph in graphs_ENZ]
graphs_ENZ_label = np.array(graphs_ENZ_label)
df2['label'] = graphs_ENZ_label

#df2.to_csv (r'graphs_ENZ.csv', index = False, header=True)


# In[36]:


Kf_ENZ = KFold(n_splits=10,shuffle = True)
Kf_ENZ.get_n_splits(df2[col])
clf_ENZ = svm.SVC(kernel='precomputed',C=0.01)
acc2=[]

for train_index, test_index in Kf_ENZ.split(df2[col]):
    
    graphs_ENZ_train, graphs_ENZ_test = df2[col].iloc[train_index], df2[col].iloc[test_index]
    graphs_ENZ_train_label, graphs_ENZ_test_label = df2["label"].iloc[train_index], df2["label"].iloc[test_index]
    
    kernel_train2 = np.dot(graphs_ENZ_train,graphs_ENZ_train.T)
    clf_ENZ.fit(kernel_train2,graphs_ENZ_train_label)
    
    kernel_test2 = np.dot(graphs_ENZ_test,graphs_ENZ_train.T)
    label_pred2 = clf_ENZ.predict(kernel_test2)
    acc2.append(accuracy_score(graphs_ENZ_test_label,label_pred2))


# In[37]:


print("The ten accuracies are as follow")
print(acc2)
print("")
print(f"The mean accuracy on the test data {np.mean(acc2)}")
print(f"The standard deviation is {np.std(acc2)}")


# In[ ]:




