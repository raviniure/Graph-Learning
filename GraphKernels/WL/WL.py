#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import networkx as nx
import data_utils as utils


# In[3]:


def get_node_colors(G):
    """
    :param G: A networkx graph G=(V,E)
    :return: A numpy array of shape (|V|, a), where a is the length of the node colors vector
    """
    colors = np.int32([node[1]["color"] for node in G.nodes(data=True)])
    return colors


# In[4]:


def get_padded_node_colors(graphs):
    """
    Computes a 3D Tensor X with shape (k, n, l) that stacks the node labels of all graphs.
    Here, k = |graphs|, n = max(|V|) and l is the number of distinct node labels.
    Node labels are encoded as l-dimensional one-hot vectors.

    :param graphs: A list of networkx graphs
    :return: Numpy array X
    """
    node_labels = [get_node_colors(g) for g in graphs]
    all_labels = np.hstack(node_labels)
    max_label = np.max(all_labels)
    min_label = np.min(all_labels)
    label_count = max_label-min_label+1

    max_size = np.max([g.order() for g in graphs])
    n_samples = len(graphs)

    X = np.zeros((n_samples, max_size, label_count), dtype=np.float32)
    for i, g in enumerate(graphs):
        X[i, np.arange(len(g.nodes())), node_labels[i]-min_label] = 1.0

    return X


# In[6]:


def if_reassign (G,n_1,n_2):
    """
    Check in Graph G, if nodes n_1,n_2 need to be reassigned 
    
    :param G: A networkx graph G=(V,E)
    :n_1, n_2: two nodes of Graph G
    :return: if True, node_n2 need to be reassigned; if False, these two nodes don't need to be reassigned.
    """
    #Check if two nodes already have different colors. 
    if G.nodes[n_1]['color'] != G.nodes[n_2]['color']:
        return False  
        #No need to reassign
    
    #Check if the degree of two nodes are different.
    elif G.degree[n_1] != G.degree[n_2]:
        return True 
        #Their degrees are different so the multiset of colors in their neighborhoods must differ
    
    #Check if the multiset of colors in their neighborhoods are same
    else:
        ind1 = [G.nodes[n]['color'] for n in G.neighbors(n_1)]
        ind2 = [G.nodes[n]['color'] for n in G.neighbors(n_2)]
        ind1 = np.sort(np.array(ind1))
        ind2 = np.sort(np.array(ind2))
        TF = (ind1 == ind2).all()
        if TF:  
            return False
        else:
            return True
        #The multiset of colors in their neighborhoods differ, reassign node n_2


def reassign (G,iter):
    """
    Implentment color refinement.
    
    :param G: A networkx graph G=(V,E)
    :iter: 0 or 1. when iter = 0, only initialize node color with node_label
                   when iter = 1, implement one round color refinement on G.
    return reassigned G
    """
    if iter == 0:
        # |colors| is the unique number of colors of G
        colors = max(utils.get_node_labels(G))
        # For every node, initialize the color label with node_label
        for node in G.nodes:
            G.nodes[node]['color'] = G.nodes[node]['node_label']
    
    #Implement one round color refinement on G. 
    elif iter == 1:
        colors = max(get_node_colors(G))
        nodes = np.array(G.nodes)
        num = len(nodes)
        for i in range(num):
            for j in range(i, num):
                # compare a pair of nodes
                if if_reassign(G, nodes[i], nodes[j]):
                    # add a new color
                    colors = colors + 1
                    # assign newly added color to the node
                    G.nodes[nodes[j]]['color'] = colors


# In[208]:


def generate_feature_vec_WL (graphs,iter):
    """
    Given a list of graphs, generate a feature vector for each of them
    
    :param graphs: a list of graphs
    :iter：        the number of rounds of implementing color refinement 
    :return: return a dataframe. Every corresponds a feature vector
    """
    # For the first round, pass it to reassign function to initialize the node colors
    for G in graphs:
        reassign(G, 0)

    #An array contains the number of colors of the graphs 
    number_of_color =[max(get_node_colors(G)) for G in graphs]
    # The dimension of the feature vectors in round 0
    dim = max(number_of_color)                                
    
    #Form a dataframe
    vec_dim = np.arange(1,dim+1,1) # create a color labels
    vec_dim = [str(n) for n in vec_dim] # convert the color label into strings
    vec_df = pd.DataFrame(columns = vec_dim)
    #Generate a feature vector for each graph, each vector entry counts the occurence of one color in the graph in round 0
    #Then we got an initial dataframe vec_df 
    for ind in range(len(graphs)):
        cols = get_node_colors(graphs[ind])
        count = np.zeros(dim)
        for col in cols:
            count[col-1]= count[col-1]+1
        vec_df.loc[ind] = count
    
    #Implement color refinement in the round 1,2,3,4......
    for r in range(1,iter+1,1):
        for G in graphs:
            reassign(G,1)
        
        # An array contains the number of colors of each graph 
        number_of_color_r =[max(get_node_colors(G)) for G in graphs]
        # The dimension of the feature vectors in round r
        dim_r = max(number_of_color_r)
        vec_dim_r = np.arange(1,dim_r+1,1)
        vec_dim_r = [str(n) for n in vec_dim_r]
        vec_df_r = pd.DataFrame(columns = vec_dim_r)
        
        for ind in range(len(graphs)):
            cols_r = get_node_colors(graphs[ind])
            count_r = np.zeros(dim_r)
            for col in cols_r:
                count_r[col-1]= count_r[col-1]+1
            vec_df_r.loc[ind] = count_r
        #Concatenate the original dataframe with the new one 
        vec_df = pd.concat([vec_df,vec_df_r],axis = 1)
    
    #Add column 'label' to the dataframe
    labels = [utils.get_graph_label(graph) for graph in graphs]
    labels = np.array(labels)
    vec_df['label'] = labels
    col_name = list(range(1,len(vec_df.columns),1))
    col_name.append('label')
    vec_df.columns = col_name
    return vec_df


# In[ ]:




