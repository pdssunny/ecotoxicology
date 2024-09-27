# Data
Adore dataset are available at \url{https://gitlab.renkulab.io/mltox/adore/-/tree/master/data?ref_type=heads}.

##  Dependencies

Dependencies:

- python 3.8+
- pytorch >=1.2
- numpy
- sklearn
- tqdm
- rekit
- deepchem

##  How to run

`01. python GCN.py <train-dataset> `  <test-dataset>

- eg `python GCN.py  `  training-A2A.csv test-A2A.csv

`02. python GAT.py <train-dataset> `  <test-dataset>

- eg `python GAT.py  `  training-A2A.csv test-A2A.csv

`03. python MPNN.py <train-dataset> `  <test-dataset>

- eg `python MPNN.py  `  training-A2A.csv test-A2A.csv

`04. python AttentiveFP.py <train-dataset> `  <test-dataset>

- eg `python AttentiveFP.py  `  training-A2A.csv test-A2A.csv

05. FPGNN Use train.py
Args:

  - data_path : The path of input CSV file. *E.g. input.csv*
  - dataset_type : The type of dataset. *E.g. classification  or  regression*
  - save_path : The path to save output model. *E.g. model_save*
  - log_path : The path to record and save the result of training. *E.g. log*

eg `python train.py  --data_path data/test.csv  --dataset_type classification  --save_path model_save  --log_path log`

Use predict.py

Args:

  - predict_path : The path of input CSV file to predict. *E.g. input.csv*
  - result_path : The path of output CSV file. *E.g. output.csv*
  - model_path : The path of trained model. *E.g. model_save/model.pt*

eg `python predict.py  --predict_path data/test.csv  --model_path model_save/test.pt  --result_path result.csv`

 ### For training

The dataset file should be a **CSV** file with a header line and label columns. E.g.

```
SMILES,LABEL
CCNc1nc(NC(C)(C)C)nc(OC)n1,0
O=[N+]([O-])c1ccc(O)cc1C(F)(F)F,1
...
```

### For predicting

The dataset file should be a **CSV** file with a header line and without label columns. E.g.

```
SMILES
Oc1c(Br)cc(Br)cc1Br
CC(C)OP(C)(=O)OC(C)C
...
```

### 
