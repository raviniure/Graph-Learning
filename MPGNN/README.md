***Results***
-------------

Different model configurations and corresponding validation MAE.

|Model| Number of MPGNN Layer | Hidden dimension |Aggregation type | Residual connections | Epochs | Batch size | Learning rate | Val_MAE |
| ------| ------ | ------ |------| ------ | ------ |------ |------ |------ |
|No.1|3|64|Sum|Yes|70|32|2*1e-3|0.278| 
|No.2|3|32|Sum|Yes|70|32|2*1e-3|0.323|
|No.3|4|64|Sum|Yes|70|32|2*1e-3|0.282|
|No.4|5|64|Sum|Yes|70|32|2*1e-3|0.261|
|No.5|5|64|Sum|No|90|32|2*1e-3|0.237|
|**No.6**|**5**|**64**|**Sum**|**Yes**|**90**|**32**|**2*1e-3**|**0.236**|
|No.7|5|64|Mean|Yes|90|32|2*1e-3|0.328|

The mean absolute errors (MAE) that we obtained for ZINC dataset across training, validation and test datasets are summarized below.

| Model | MAE_Train | MAE_Val | MAE_Test |
| ------ | ------- |------- | ------ | 
|No.1|0.231|0.278|0.281|
|No.2|0.288|0.323|0.346|
|No.3|0.225|0.282|0.284|
|No.4|0.236|0.261|0.260|
|No.5|0.191|0.237|0.227|
|**No.6**|**0.175**|**0.236**|**0.224**|
|No.7|0.269|0.328|0.366|

Overall, updating hidden units to 64 significantly ouperforms the models with hidden units of 32. The effects can be seen in between model 1 and model 2. Additionally, introducing an additional layer of MPGNN layer improved the performance of the model, for example in cases of model 1, 3 and 4 (3-MPGNN vs 4-MPGNN vs 5-MPGNN). When it came to residual connections, model with residual connections seems to converge faster than the models without residual connection which is an expected behavior due to easy passage of gradients. Ultimately, the model (no. 6) with 64 hidden units, 5 MPGNN layers, aggregation type as sum, and with residual connection achieved best results for us when the model was trained for 90 epochs with 2 times the default learning rate (1e-3) and batch size of 32.


****Datasets****
----------------
This directory contains the ZINC_Train, ZINC_Val and ZINC_Test datasets.

****Scripts****
----------------
"function.py" contains the implementation for computing Incidence Matrices and some other functions for later padding the matrices.

"layers.py" contains the implementation of Message Passing GNN Layer.

"train_model.py" performs training, validation and evaluation on ZINC dataset.



****Execution****
-----------------

 **Note**
These steps **should only** be executed from **MPGNN** directory.

For the model which performs best on validation set
    
```sh
python ./train_model.py -weighted=0 -num=5 -rc=1 -units=64 -epoch=90
```

