***Results***
-------------

The accuracy and standard deviation that we obtained for **Node Classification** are summarized below.

| Dataset | p=1, q=1 | p=0.1, q=1 | p=1, q=0.1| epochs
| ------ | ------ |------| ------ | ------ | 
| Cora |  0.80 (+/-0.027) | 0.77 (+/-0.018) | 0.78 (+/-0.021) | 5 |
| Citeseer |  0.60 (+/-0.014) | 0.58 (+/-0.023)| 0.59 (+/-0.027) | 20 |

The accuracy and roc-auc score that we obtained for **Link Prediction** are summarized below.
| Dataset | mean accuracy | ROC-AUC score |
| ------ | ------ |------| 
| PPI |  0.74 (+/-0.003) | 0.81 (+/-0.004) | 
| Facebook |  0.92 (+/-0.001) | 0.97 (+/-0.001) |  


****Datasets****
----------------
This directory contains the Cora and Citeseer datasets for node classification and the Facebook and PPI datasets for link prediction.

****Scripts****
----------------
"randomwalk.py" contains the implementation for randomwalk. 

"node2vec.py" contains the implementation of computing a node2vec embedding for a given graph G.

"NodeClassification.py" performs node classification on datasets Cora and Citeseer

"LinkPrediction.py" performs link prediction on datasets PPI and Facebook

****Execution****
-----------------

 **Note**
These steps **should only** be executed from **NodeEmbeddings** directory.

    
For task 3, Cora dataset:
    
```sh
python ./NodeClassification.py -data=./datasets/Cora/data.pkl -epoch=5
```

For task 3, Citeseer dataset:

```sh
python ./NodeClassification.py -data=./datasets/Citeseer/data.pkl -epoch=20
```

For task 4, PPI dataset :
    
```sh
python ./LinkPrediction.py -data=./datasets/PPI/data.pkl
```

For task 4, Facebook dataset (takes about 42 minutes):

```sh
python ./LinkPrediction.py -data=./datasets/Facebook/data.pkl
```

