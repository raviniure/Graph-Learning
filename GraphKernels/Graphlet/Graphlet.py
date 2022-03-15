import networkx as nx
import numpy as np
import random

# Function graph_atlas_g() return the list of all graphs with up to seven nodes.
g7=nx.graph_atlas_g()

# Extract all the five_node graphs from g7
five_node_graphlet = []
for graph in g7:
    if nx.number_of_nodes(graph) == 5:
        five_node_graphlet.append(graph)

# Given a graph and graphlet_list, generate the feature vector of the graph
def generate_feature_vec (G, graphlet_list):
    
    # For counting how often each graphlet occurs
    count = np.zeros(34)
    
    # Repeat the sampling 1000 times
    for r in range(1000):
        # Sample 5 nodes from G and generate its induced subgraph
        sampled_nodes = random.sample(G.nodes,5)
        sampled_graph = G.subgraph(sampled_nodes)
        
        # Check which graphlet the induced subgraph is isomorphic to
        for i in range(34):
            if nx.is_isomorphic(graphlet_list[i], sampled_graph) == True:
                count[i] = int(count[i]+1)
                break
        r=r+1
    return count
