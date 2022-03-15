#!/usr/bin/env python
# coding: utf-8

# In[4]:


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


# In[5]:


##################################################################
###Use WL kernel to train Support Vector Machines on datasetsDD###
##################################################################

#import data
with open('datasets/datasets/DD/data.pkl', 'rb') as f:
    graphs_DD = pickle.load(f)

#This step takes about 16 minutes
DDdf = generate_feature_vec_WL(graphs_DD,4) 


# In[6]:


X_DD_WL=list(DDdf.columns)
del(X_DD_WL[-1])

Kf_DD_WL = KFold(n_splits=10,shuffle = True)
Kf_DD_WL.get_n_splits(DDdf[X_DD_WL])
clf_DD_WL = svm.SVC(kernel='precomputed',C = 0.01)

acc_DD_WL=[]

for train_index, test_index in Kf_DD_WL.split(DDdf[X_DD_WL]):
    
    graphs_train, graphs_test = DDdf[X_DD_WL].iloc[train_index], DDdf[X_DD_WL].iloc[test_index]
    graphs_train_label, graphs_test_label = DDdf["label"].iloc[train_index], DDdf["label"].iloc[test_index]
    
    kernel_train = np.dot(graphs_train,graphs_train.T)
    clf_DD_WL.fit(kernel_train,graphs_train_label)
    
    #on test data
    kernel_test = np.dot(graphs_test,graphs_train.T)
    label_pred = clf_DD_WL.predict(kernel_test)
    #Calculte accuracy
    acc_DD_WL.append(accuracy_score(graphs_test_label,label_pred))


# In[7]:


print("The ten accuracies are as follow")
print(acc_DD_WL)
print("")
print(f"The mean accuracy on the test data {np.mean(acc_DD_WL)}")
print(f"The standard deviation is {np.std(acc_DD_WL)}")


# In[ ]:




