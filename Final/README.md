***Results***
-------------

The metrics that we obtained for the 4 datasets are summarized below.

| Dataset | ROC-AUC (Train) | ROC-AUC (Validation) | Accuracy (Train) | Accuracy (Validation) |
| ------ | ------ | ------ | ------ | ------ | 
| HIV | 0.890 | 0.791 | - | - |
| Mutagenicity | - | - | 0.834 | 0.813 |
| IMDB |  - | - |0.710 |0.80 |
| Amazon |  - | - | 0.958 | 0.921 |


****Datasets****
----------------
This directory contains the HIV, Mutagenicity, IMDB and Amazon datasets.

****Scripts****
----------------
"HIV.py" contains the training for HIV dataset

"Mutagenicity.py" contains the training for Mutagenicity dataset

"imdb.py" contains the implementation of Weisfeiler-Leman-Kernel and training for IMDB dataset

"Amazon.py" contains the training for Amazon dataset

"functions_layers.py" contains the functions and classes of layers for implementing MPGNN.

"gcn.py" contains the functions and classes of layers for implementing GCN.

****Solution****
----------------

**Dataset HIV**

The model used for graph classification consists of 4 MPGNN Layers, a pooling layer, a drop out layer and dense layers. The layers are with residual connection and the aggregation function is SUM.
 
The node embedding and edges features are initialized with node features and edge features of graphs respectively. Before training, the node & edge features & Incidence matrices of training set and evaluation set must be padded to achieve a uniform size.

**Dataset Mutagenicity**

The model used for graph classification consists of 4 GCN Layers, a pooling layer, a dropout layer and dense layers. The node embedding is initialized with with one-hot encoding of the node label. The normalized adjacency matrices and the node features of training set and evaluation set should padded to achieve a uniform size.

**Dataset IMDB**

The model uses Weisfeiler-Leman kernel with four rounds of color refinement to obtain the features. Once the features are obtained, the gram matrix is calculated. SVM with precomputed kernel and trade-off parameter of C = 100 is used to perform the binary classification task. 

**Dataset Amazon**

A 2-layer GCN Model is adopted for the node classification. The output units of the hidden layer is 160 and the activation function is ReLU. The output layer has a dimension of 8 with Softmax activation function.

The node embedding is initialized with the node features. The preliminary step is to pad the normalized adjacency matrices and the node features of training set and evaluation set to achieve a uniform size. Then they can be used as the inputs of the model. 


****Execution****
-----------------

 **Note**
These steps **should only** be executed from **Competition** directory.

For HIV
    
```sh
python main.py --dataset=HIV --eval_path=./datasets/HIV/HIV_Val/data.pkl  
```

For Mutagenicity

```sh
python main.py --dataset=Mutagenicity --eval_path=./datasets/Mutagenicity/Mutagenicity_Val/data.pkl
```

For IMDB

```sh
python main.py --dataset=IMDB --eval_path=./datasets/IMDB/IMDB_Val/data.pkl 
```

For Amazon
    
```sh
python main.py --dataset=Amazon --eval_path=./datasets/Amazon/Amazon_Val/data.pkl
```

    
