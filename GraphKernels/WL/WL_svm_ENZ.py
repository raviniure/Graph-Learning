#!/usr/bin/env python
# coding: utf-8


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


# In[18]:


####################################################################
###Use WL kernel to train Support Vector Machines on datasets ENZ###
####################################################################

#import data
with open('datasets/datasets/ENZYMES/data.pkl', 'rb') as f:
    graphs_ENZ = pickle.load(f)

ENZdf = generate_feature_vec_WL(graphs_ENZ,4)




X_ENZ_WL=list(ENZdf.columns)
del(X_ENZ_WL[-1])

Kf_ENZ_WL = KFold(n_splits=10,shuffle = True)
Kf_ENZ_WL.get_n_splits(ENZdf[X_ENZ_WL])
clf_ENZ_WL = svm.SVC(kernel='precomputed',C = 0.01)

acc_ENZ_WL=[]


for train_index, test_index in Kf_ENZ_WL.split(ENZdf[X_ENZ_WL]):
    
    graphs_train, graphs_test = ENZdf[X_ENZ_WL].iloc[train_index], ENZdf[X_ENZ_WL].iloc[test_index]
    graphs_train_label, graphs_test_label = ENZdf["label"].iloc[train_index], ENZdf["label"].iloc[test_index]
    
    kernel_train = np.dot(graphs_train,graphs_train.T)
    clf_ENZ_WL.fit(kernel_train,graphs_train_label)
    
    #on test data
    kernel_test = np.dot(graphs_test,graphs_train.T)
    label_pred = clf_ENZ_WL.predict(kernel_test)

    #Calculte accuracy
    acc_ENZ_WL.append(accuracy_score(graphs_test_label,label_pred))


# In[26]:


print("The ten accuracies are as follow")
print(acc_ENZ_WL)
print("")
print(f"The mean accuracy on the test data {np.mean(acc_ENZ_WL)}")
print(f"The standard deviation is {np.std(acc_ENZ_WL)}")


# In[ ]:




