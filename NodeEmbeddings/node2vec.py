import numpy as np
import networkx as nx
import tensorflow as tf
from tqdm import tqdm
from randomwalk import p_randomwalk, get_negative_sample


def generate_W(graph, p, q, walk_length, num_walks):
    """
    Function to generate walk matrix with dimension |num nodes| x |number of walks=5| x |walk length=5|
    :param graph: networkx graph object
    :param p: Breadth First Search (BFS) parameter
    :param q: Depth First Search (DFS) parameter
    :param walk_length: value indicating number of walks for each start node
    :param num_walks: value indicating length of random walks from each start node
    :return: set of numpy arrays
            input_start_node - |num nodes| x 1 indicating start nodes
            input_sample_walk - |num nodes| x |number of walks=5| x |walk length=5| indicating arrays of positive walks
            input_negative_sample - |num nodes| x |number of walks=5| x |walk length=5| indicating arrays of neg. walks
    """
    V = len(graph.nodes)

    # Initialize input_start_node, input_sample_walk, input_negative_sample
    input_start_node = np.reshape(np.array(graph.nodes), (V, 1))  # Reshape indices shape = (|V|,1)
    input_sample_walk = np.zeros(shape=(V, num_walks, walk_length))
    input_negative_sample = np.zeros(shape=(V, num_walks, walk_length))

    nodes = list(graph.nodes)

    for node in nodes:
        start = node
        for i in range(num_walks):
            w = p_randomwalk(graph, start, p, q, walk_length)
            input_sample_walk[node - 1][i] = w
            input_negative_sample[node - 1][i] = get_negative_sample(graph, w)

    return input_start_node, input_sample_walk, input_negative_sample


class Node2Vec:

    def __init__(self, graph: nx.Graph, p, q, emb_dimensions: int = 128, walk_length: int = 5, num_walks: int = 5):
        """
        Initiates the Node2Vec object.
        :param graph: Input graph
        :param emb_dimensions: Embedding dimensions (default: 128)
        :param walk_length: Number of nodes in each walk (default: 5)
        :param num_walks: Number of walks per node (default: 5)
        :param p: probability of returning to source node or BFS parameter
        :param q: probability of moving to a node away from the source node or DFS parameter
        """
        self.graph = graph
        self.emb_dimensions = emb_dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        # Number of nodes |V|
        self.V = len(self.graph.nodes)

    def loss_function(self, start_node_emb, sample_walk_emb, negative_sample_emb):
        """
        Function that calculates loss value for each node embedding
        :param start_node_emb: Tensor embedding for start node
        :param sample_walk_emb: Tensor embeddings for positive sample walks with
                                |batch size| x |num walks| x |walk length| x |embedding dimension|
        :param negative_sample_emb: Tensor embeddings for negative sample walks with
                                    |batch size| x |num walks| x |walk length| x |embedding dimension|
        :return: loss array with |batch size| x 1
        """

        # Each row, corresponding to embedding of starting node, of the X matrix is dotted with
        # node embedding of each node in a positive sample of walk.
        # shape = (batch_size, number_of_walks=5, walk_length=5)
        x_mul = tf.keras.layers.Dot(axes=(1, 3))([start_node_emb, sample_walk_emb])

        # Represents dot product of embedding vector of each node and embedded matrix of walks from
        # the corresponding node.
        # shape = (batch_size, )
        loss1 = tf.reduce_sum(x_mul, [1, 2])

        # Calculate the second portion of the loss
        # Union set of positive walk sample and negative walk sample
        w_union = tf.keras.layers.Concatenate(axis=2)([sample_walk_emb, negative_sample_emb])

        # Each row, corresponding to embedding of starting node, of the X matrix is dotted with
        # node embedding of each node in a the union set of positive and negative walk.
        # shape = (batch_size, number_of_walks=5, walk_length(+ve and -ve combined)=10)
        x2 = tf.keras.layers.Dot(axes=(1, 3))([start_node_emb, w_union])

        # Denominator part of the P(v|s)
        # shape = (batch_size, number_of_walks=5)
        x3 = tf.reduce_logsumexp(x2, 2)

        # Summing across all the walks starting from the same node
        # shape = (batch_size, )
        loss2 = self.num_walks * tf.reduce_sum(x3, 1)

        # This essentially means Probability of getting a node 'v' given start node as 's'
        # shape = (batch_size, )
        loss = loss2 - loss1
        return loss

    def create_model(self):
        """
        Function that lays out different layers and returns a model
        :return: keras.Model object
        """

        # Input layer definition
        inputs = {
            "start_node": tf.keras.layers.Input(name="start_node", shape=()),
            "sample_walk": tf.keras.layers.Input(name="sample_walk", shape=(5, 5,)),
            "negative_sample": tf.keras.layers.Input(name="negative_sample", shape=(5, 5,))
        }

        # Initialize embedding layer
        emb_layer = tf.keras.layers.Embedding(
            input_dim=self.V,
            output_dim=self.emb_dimensions,
            embeddings_initializer="he_normal", name="embedding", trainable=True)

        # Look up embedding
        start_node_emb = emb_layer(inputs["start_node"])
        sample_walk_emb = emb_layer(inputs["sample_walk"])
        negative_sample_emb = emb_layer(inputs['negative_sample'])

        # Loss function
        loss = self.loss_function(start_node_emb, sample_walk_emb, negative_sample_emb)

        # Create the model
        model = tf.keras.Model(inputs=inputs, outputs=loss)

        return model


def train_X(graph, p, q, epochs=5):
    """
    Function that takes a graph, BFS and DFS parameter, and epoch value and returns node embeddings
    :param graph: networkx graph object
    :param p: BFS parameter
    :param q: DFS parameter
    :param epochs: Number of epochs to be trained to obtain node embeddings
    :return: Weights or node embeddings with |num nodes| x |Embedding dimension|
    """

    # Given batch_size
    batch_size = 128

    walk_length = 5
    num_walks = 5

    # Number of nodes |V|
    V = len(graph.nodes)

    # Generate positive and negative walk tensors
    start_node_input, input_walk, input_negative = generate_W(graph, p, q, walk_length, num_walks)

    # Initialize inputs
    input_graph = {"start_node": start_node_input - 1, "sample_walk": input_walk - 1,
                   "negative_sample": input_negative - 1}

    # Build the Model
    node2vec = Node2Vec(graph, p, q)
    model = node2vec.create_model()

    #
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    loss_function = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.SUM)

    # Train X with SGD
    for i in range(epochs):
        for idx in tqdm(range(0, V, batch_size)):
            # Create input dictionary with start node, positive walk, and negative walk of batch_size
            input_graph = {"start_node": (start_node_input - 1)[idx:idx + batch_size],
                           "sample_walk": (input_walk - 1)[idx:idx + batch_size],
                           "negative_sample": (input_negative - 1)[idx:idx + batch_size]}

            # Aiming to have total loss equal to 0; optimizing towards that goal
            y = np.zeros(shape=(1,))

            # Using gradient tape for training embeddings
            with tf.GradientTape() as tape:
                logits = model(input_graph)
                loss_value = loss_function(y, logits)
            gradients = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    # Get the matrix X
    weights = model.get_layer('embedding').get_weights()[0]
    return weights
