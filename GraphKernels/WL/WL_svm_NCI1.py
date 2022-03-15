#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import networkx as nx
import pickle
import pandas as pd

from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.metrics import accuracy_score

import data_utils as utils
from WL import get_node_colors
from WL import if_reassign
from WL import reassign
from WL import generate_feature_vec_WL


# In[2]:


#####################################################################
###Use WL kernel to train Support Vector Machines on datasets NCI1###
#####################################################################

#import data
with open('datasets/datasets/NCI1/data.pkl', 'rb') as f:
    graphs_NCI1 = pickle.load(f)

NCI1df = generate_feature_vec_WL(graphs_NCI1,4)


# In[4]:


X_NC_WL=list(NCI1df.columns)
del(X_NC_WL[-1])

Kf_NC_WL = KFold(n_splits=10,shuffle = True)
Kf_NC_WL.get_n_splits(NCI1df[X_NC_WL])
clf_NC_WL = svm.SVC(kernel='precomputed',C=0.01)

acc_NC_WL=[]


for train_index, test_index in Kf_NC_WL.split(NCI1df[X_NC_WL]):
    
    graphs_train, graphs_test = NCI1df[X_NC_WL].iloc[train_index], NCI1df[X_NC_WL].iloc[test_index]
    graphs_train_label, graphs_test_label = NCI1df["label"].iloc[train_index], NCI1df["label"].iloc[test_index]
    
    kernel_train = np.dot(graphs_train,graphs_train.T)
    clf_NC_WL.fit(kernel_train,graphs_train_label)
    
    #on test data
    kernel_test = np.dot(graphs_test,graphs_train.T)
    label_pred = clf_NC_WL.predict(kernel_test)
    #Calculte accuracy
    acc_NC_WL.append(accuracy_score(graphs_test_label,label_pred))


# In[5]:


print("The ten accuracies are as follow")
print(acc_NC_WL)
print("")
print(f"The mean accuracy on the test data {np.mean(acc_NC_WL)}")
print(f"The standard deviation is {np.std(acc_NC_WL)}")


# In[ ]:




