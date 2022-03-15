import numpy as np
import pandas as pd
from argparse import ArgumentParser
import pickle

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from data_utils import get_node_labels, get_graph_label, get_padded_node_labels
from randomwalk import is_valid_file, p_randomwalk, get_negative_sample
from node2vec import generate_W, Node2Vec,train_X


def NodeClassification(graph, p, q, epochs):
    """
    Function that performs node classification task
    :param graph: networkx graph object
    :param p: BFS parameter
    :param q: DFS parameter
    :param epochs: number of epochs that should be run for training embeddings
    :return: none
    """
    batch_size = 128
    
    # Get the trained embedding matrix x
    weights = train_X(graph, p, q, epochs)
        
    # Form a dataframe for node classification
    df = pd.DataFrame(data=weights)
    df['label'] = get_node_labels(graph)

    clf = LogisticRegression()
    ind = list(np.arange(0, batch_size, 1))

    Kf = StratifiedKFold(n_splits=10, shuffle=True)
    Kf.get_n_splits(df['label'])

    acc = []
    for train_index, test_index in Kf.split(df[ind], df['label']):
        train, test = df[ind].iloc[train_index], df[ind].iloc[test_index]
        train_label, test_label = df["label"].iloc[train_index], df["label"].iloc[test_index]

        clf.fit(train, train_label)

        # Test data
        label_pred = clf.predict(test)

        # Calculate accuracy
        acc.append(accuracy_score(test_label, label_pred))

    print("***Statistics on Test Data***")
    print(f"Mean accuracy:  {np.mean(acc):0.04f}")
    print(f"Standard deviation: {np.std(acc):0.04f}")




if __name__ == "__main__":
    """
    Command liner parser initialization
    """
    parser = ArgumentParser(description="Loading datasets")
    parser.add_argument("-data", dest="data", required=True,
                        help="input file with graphs dataset", metavar="FILE",
                        type=lambda x: is_valid_file(parser, x))
    parser.add_argument("-epoch", dest="epoch", required=True,
                        help="number of epochs for training", metavar="INT",
                        type=int)

    args = parser.parse_args()

    """
    Load Dataset
    """
   
    epoch = args.epoch
    graph = pickle.load(args.data)[0]
    
    NodeClassification(graph=graph,p=1,q=1,epochs = epoch)
    NodeClassification(graph=graph,p=0.1,q=1,epochs = epoch)
    NodeClassification(graph=graph,p=1,q=0.1,epochs = epoch)
