***Results***
-------------

The results that we obtained with various kernels are summarized below.

| Datasets | Graphlet Kernel | Closed Walk Kernel | Weisfeiler-Lehman Kernel |
| ------ | ------ |------| ------ | 
| DD | 0.74 (+/- 0.042) |0.74 (+/- 0.029)| 0.78 (+/- 0.032) | 
| ENZYMES | 0.27 (+/- 0.059) |0.20 (+/- 0.037)| 0.30 (+/- 0.044) | 
| NCI1 | 0.63 (+/- 0.021) |0.63 (+/- 0.025)| 0.64 (+/- 0.023) | 

_Comparison of results to `http://people.mpi-inf.mpg.de/~mehlhorn/ftp/genWLpaper.pdf`_

***DD Dataset***: In the paper, the highest mean accuracy obtained is approximately 79 % with standard deviation of about +/- 0.36. This mean accuracy is obtained with WL subtree algorithm. In our case, we obtained 78% (+/- 0.32) - a comparable accuracy rate with our implementation of the WL Kernel. Both of our graphlet and closed walk kernel implementation also came close to the best performance, about 5% lower than the best. 


***ENZYMES Dataset***: In the paper, the highest mean accuracy obtained is approximately 59% (+/- 1.05) with shortest path version of WL kernel. In our case, we obtained much lower 30% (+/- 0.44) mean accuracy with our implementation of WL kernel. As expected, the other kernels, graphlet and closed walk, did not perform well on the ENZYMES dataset.


***NCI1 Dataset***: In the paper, the highest mean accuracy obtained is about 84% (+/- 0.36) with shortest path version of WL kernel. We didn't come close to such performance even with our best kernel, WL kernel, predicting accurately about 64% (+/- 0.23) percent on the dataset. However, with graphlet and closed walk kernel, we were able to obtain comparable results to the author's implementation of graphlet and closed walk kernel at about 63% (+/- 0.2).   



***Directory Structure***
-----------------------
This directory contains implementations of all the Graph Kernels. It also hosts the dataset that was provided for the
first exercise.

****CWKL****
------------

- This directory is for implementation of __Closed Walk Kernel__ and SVMs on various dataset.
    
    __Execution__
    
    **Note**
        These steps **should only** be executed from GraphKernels directory.

    For DD dataset:
    
    ```sh
    $ python ./CWKL/svm_DD.py
    ```

    For ENZ dataset:

    ```sh
    $ python ./CWKL/svm_ENZ.py
    ```

    For NCI1 dataset: 

    ```sh
    $ python ./CWKL/svm_NCI1.py
    ```
- **Justification for Walk Length**

    We explored different walk lengths for different dataset and settled at l = 5. The three things that we considered for this are: overall structure of graphs, computational efficiency, and impact on accuracy. First of all, the graphs which were connected had on average the diameter of less than 20 for all datasets. As such, it was not deemed necessary to look for longer walk lengths. Secondly, and the most important aspect is that, with longer walks, the feature space was getting bigger and higher values in higher space were dominating. Additionally, with longer walks, the computation was getting expensive without much impact on accuracy. We found the accuracy started saturating around l = 5 and hence used the walk length as l = 5. 


****Graphlet****
----------------
- This directory is for implementation of __Graphlet Kernel__ and SVMs on various dataset.
    
    __Execution__

    **Note**
        These steps **should only** be executed from GraphKernels directory.

    For implementation of Graphlet Kernel:

    ```sh
    $ python ./Graphlet/Graphlet.py
    ```

        
    For DD dataset:
    
    ```sh
    $ python ./Graphlet/Graphlet_svm_DD.py
    ```

    For ENZ dataset:

    ```sh
    $ python ./Graphlet/Graphlet_svm_ENZ.py
    ```

    For NCI1 dataset: 

    ```sh
    $ python ./Graphlet/Graphlet_svm_NCI1.py
    ```



****WL****
----------------
- This directory is for implementation of __Weisfeiler Lehman Kernel__ and SVMs on various dataset.
    
    __Execution__

    **Note**
        These steps **should only** be executed from GraphKernels directory.
    
    For implementation of WL Kernel:

    ```sh
    $ python ./WL/WL.py
    ```
        
    For DD dataset:
    
    ```sh
    $ python ./WL/WL_svm_DD.py
    ```

    For ENZ dataset:

    ```sh
    $ python ./WL/WL_svm_ENZ.py
    ```

    For NCI1 dataset: 

    ```sh
    $ python ./WL/WL_svm_NCI1.py
    ```


****datasets****
----------------
- This directory contains DD, ENZYMES, and NCI1 dataset used to evaluate various kernels.

****Sandbox****
---------------
- This directory contains dirty code. ** Do Not Look **





