import numpy as np
import networkx as nx
from argparse import ArgumentParser
import pickle
import random

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from randomwalk import is_valid_file, p_randomwalk, get_negative_sample
from node2vec import generate_W, Node2Vec, train_X


def generate_edge_emb(EdgeSet, node_emd):
    """
    Function for turning the node embedding into an edge embedding.
    Given a set of edges of a graph and its trained node_embedding matrix,
    return its edge embedding matrix.

    :param EdgeSet: list of edges
    :param node_emd: corresponding node embeddings
    :return mat: |num of edges| x |128| dimension matrix that holds edge embeddings
    """

    l = len(EdgeSet)

    # Initialize the matrix that holds embeddings
    mat = np.zeros((l, 128))
    for e in range(l):
        # pick one edge
        edge = EdgeSet[e]
        n1 = edge[0] - 1
        n2 = edge[1] - 1

        # Calculate the Hadamard Product
        for j in range(128):
            mat[e][j] = node_emd[n1][j] * node_emd[n2][j]

    return mat


if __name__ == "__main__":
    # List to hold accuracy scores
    accuracy = []
    sco = []

    """Repeat the whole process for 5 times:
        1. sample E_eval->get E_train and E_eval
        2. sample N_train and N_eval
        3. compute a node2vec embedding for G_train
        4. turn node embedding into edge embedding
        5. train a classifier and evaluate
    """

    for i in range(5):
        """Command liner parser initialization"""
        parser = ArgumentParser(description="Loading datasets")
        parser.add_argument("-data", dest="data", required=True,
                            help="input file with graphs dataset", metavar="FILE",
                            type=lambda x: is_valid_file(parser, x))
        args = parser.parse_args()

        """Load Dataset"""
        Graph = pickle.load(args.data)[0]

        # Obtain list of edges
        E = list(Graph.edges())

        # Number of edges in Graph
        num_edges = Graph.number_of_edges()

        # 20% of the edges are to be considered for evaluation
        num_eval_edges = int(num_edges * 0.2)  # Size of E_eval

        # Number of connected components
        G_comp = nx.number_connected_components(Graph)

        # Obtain graph complement
        G_complement = nx.complement(Graph)  # G_complement corresponds to graph (V, (V*V)\E)
        E_complement = list(G_complement.edges())  # Set of edges (V*V)\E

        """Step1: Sample E_eval and return G_train and E_eval"""
        E_eval = []

        # Make copy of G and E
        G_train = Graph
        Edges = E
        print(len(Edges))

        for i in range(num_edges):
            num = len(E_eval)  # Record the number of elements in E_eval
            if num == num_eval_edges:  # Get enough number of samples, break
                break
            else:
                edge = random.sample(Edges, 1)[0]  # Sample an edge from graph randomly
                G_train.remove_edge(*edge)  # Remove this edge from G
                Edges.remove(edge)  # Update Edges

                if nx.number_connected_components(G_train) == G_comp:
                    # Number of components doesn't increase, the edge can be sampled into E_eval
                    num = num + 1
                    E_eval.append(edge)
                else:
                    # Number of components increase, the edge can't be sampled into E_eval so add it back to G
                    G_train.add_edge(*edge)

        # E_train = E\E_eval
        E_train = list(G_train.edges())
        num_train_edges = len(E_train)

        """Step2: Sample N_train from (V*V)\E"""
        N_train = random.sample(E_complement, num_train_edges)
        N_eval = random.sample(list(set(E_complement).difference(set(N_train))), num_eval_edges)

        """Step3: Compute the node2vec embedding matrix for G_train = ()
        
        There might be few isolated nodes which can cause problem while perform random walk
        since they don't have any neighbours.Therefore we add a self loop on such nodes.
        It won't have much impact on result."""

        for node in list(G_train.nodes()):
            neigh = list(nx.neighbors(G_train, node))
            if len(neigh) == 0:  # Check if the set of neighbours of node is empty
                G_train.add_edge(node, node)  # Add a self loop on it

        # Compute a node2vec embedding for G_train and get the node embedding matrix
        X_node_emb = train_X(G_train, 1, 1, 10)

        """Step4: The edge embedding of E_train and N_train"""
        E_train_emd = generate_edge_emb(E_train, X_node_emb)
        N_train_emd = generate_edge_emb(N_train, X_node_emb)
        data_train = np.concatenate((E_train_emd, N_train_emd), axis=0)

        # The edge embedding of E_eval and N_eval
        E_eval_emd = generate_edge_emb(E_eval, X_node_emb)
        N_eval_emd = generate_edge_emb(N_eval, X_node_emb)
        data_eval = np.concatenate((E_eval_emd, N_eval_emd), axis=0)
        #print(data_eval.shape)

        # The label of the edge
        y_train = np.concatenate((np.ones(num_train_edges), np.zeros(num_train_edges)), axis=None)
        y_eval = np.concatenate((np.ones(num_eval_edges), np.zeros(num_eval_edges)), axis=None)

        """Step5"""
        clf = LogisticRegression()
        clf.fit(data_train, y_train)  # Fit the model
        label_pred = clf.predict(data_eval)  # Evaluate the model
        acc = accuracy_score(y_eval, label_pred)
        ra = roc_auc_score(y_eval, clf.predict_proba(data_eval)[:, 1])
        print(f"The accuracy in this round is {acc}")
        print(f"The ROC-AUC score in this round is {ra}")
        accuracy.append(acc)
        sco.append(ra)

    # Display results
    print(f"The mean accuracy is {np.mean(accuracy)}")
    print(f"The standard deviation of accuracy is {np.std(accuracy)}")
    print(f"The mean ROC-AUC score is {np.mean(sco)}")
    print(f"The standard deviation of ROC-AUC score is {np.std(sco)}")
