***Results***
-------------

The accuracy and standard deviation that we obtained for **Node Classification** are summarized below.

| Datasets | Training | Testing |
| ------ | ------ |------|
| Cora | 0.90 (+/- 0.0035) | 0.79 (+/- 0.128)| 
| Citeseer | 0.91 (+/- 0.0023) | 0.73 (+/- 0.0052)| 

For training we used **cross-entropy** loss with **Adam** optimizer. 

**Dataset Cora** and **Dataset Citeseer:**: with default learning rate and the number of epochs is 80.

The accuracy and standard deviation that we obtained for **Graph Classification** are summarized below.

| Datasets | Training | Testing |
| ------ | ------ |------|
| NCI1 |  0.78(+/- 0.0059) | 0.76(+/- 0.0266)| 
| ENZYMES |  0.76(+/-0.0449 ) | 0.56(+/-0.0553 )| 

For training we used **cross-entropy** loss with **Adam** optimizer. 

**Dataset NCI1:** The learning rate is 1e-3, the number of epochs is 25. We added a dropout layer after the hidden dense layer with dropout rate 0.15.

**Dataset ENZYMES:** The learning rate is 1e-3, the number of epochs is 100.  We added a dropout layer after the hidden dense layer with dropout rate 0.10.

****Datasets****
----------------
This directory contains Citeseer_Eval, Citeseer_Train, Cora_Eval, Cora_Train, NCI1 and ENZYMES used to evaluate Node Classification and Graph Classification.

****Scripts****
----------------
"node_class.py" contains implementation for node classification and evaluation. 
"graph_class.py" contains implementation for graph classificiation and evaluation.

****Execution****
-----------------

 **Note**
These steps **should only** be executed from **GraphNeuralNetworks** directory.

    
For task 1 and 2, Citeseer dataset:
    
```sh
python ./node_class.py -train=./datasets/Citeseer_train/data.pkl -test=./datasets/Citeseer_eval/data.pkl
```

For task 1 and 2, Cora dataset:

```sh
python ./node_class.py -train=./datasets/Cora_train/data.pkl -test=./datasets/Cora_eval/data.pkl
```

For task 3 and 4, ENZYMES dataset:
    
```sh
python ./graph_class.py -data=./datasets/ENZYMES/data.pkl -epoch=100 -drop=0.10 
```

For task 3 and 4, NCI1 dataset:

```sh
python ./graph_class.py -data=./datasets/NCI1/data.pkl -epoch=25 -drop=0.15
```
