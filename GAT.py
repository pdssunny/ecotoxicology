import deepchem as dc
from deepchem.models import GATModel
import numpy as np
import math
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, recall_score
import sys
import csv
import os

def score(dataset):
    y_true = []
    for x, y, w, id in dataset.itersamples():
        y_true.append(y)
    y_pred = model.predict(dataset)
    y_pre = y_pred[:,1]
    auc_roc_score = roc_auc_score(np.array(y_true).squeeze(), y_pre)
    y_pred_print = np.argmax(y_pred, axis=-1)
    tn, fp, fn, tp = confusion_matrix(np.array(y_true).squeeze(), y_pred_print.squeeze()).ravel()
    se = tp / (tp + fn)
    sp = tn / (tn + fp)  
    q = (tp + tn) / (tp + fn + tn + fp)
    mcc = (tp * tn - fn * fp) / math.sqrt((tp + fn) * (tp + fp) * (tn + fn) * (tn + fp))
    P = tp / (tp + fp)
    F1 = (P * se * 2) / (P + se)
    BA = (se + sp) / 2
    return tp, tn, fn, fp, se, sp, mcc, q, auc_roc_score, F1, BA

def model_builder(model_dir, **model_params):
    model = GATModel(mode='classification',n_tasks=len(tasks),**model_params)
    return model

train_file = sys.argv[1]
test_file =  sys.argv[2]

tasks = ['LABEL']
featurizer = dc.feat.MolGraphConvFeaturizer()
loader = dc.data.CSVLoader(tasks=tasks, feature_field="SMILES", featurizer=featurizer)
all_train_dataset = loader.create_dataset(train_file, shard_size=8192)
transformers = []
# # data split
splitter = dc.splits.RandomStratifiedSplitter()
train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset=all_train_dataset,
                                                                             frac_train=0.9,
                                                                             frac_valid=0.1,
                                                                             frac_test=0,
                                                                             seed=1231)
test_dataset = loader.create_dataset(test_file, shard_size=8192)

batch_size = 64
num_epochs = 30
best_roc_auc_score = 0
valid_true=valid_dataset.y

metric = dc.metrics.Metric(dc.metrics.roc_auc_score) #AUC来调参
params_dict={'dropout':[0.1,0.3,0.5],'weight_decay':[0.0001,0.000001],'learning_rate':[0.1, 0.01, 0.001],
            'n_attention_heads':[8, 16, 32]}



optimizer=dc.hyper.GridHyperparamOpt(model_builder)
best, best_hyperparams,all_results= optimizer.hyperparam_search(params_dict,
                                                                train_dataset,
                                                                valid_dataset,
                                                                metric)
model = GATModel(mode='classification',
                 n_tasks=len(tasks),
                 batch_size=batch_size,
                 dropout=best_hyperparams[0],
                 weight_decay = best_hyperparams[1],   
                 learning_rate = best_hyperparams[2],
                 dn_attention_heads = best_hyperparams[3])

tp1, tn1, fn1, fp1, se1, sp1, mcc1, q1, auc_roc_score1, F1_1, BA1 = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
tp2, tn2, fn2, fp2, se2, sp2, mcc2, q2, auc_roc_score2, F1_2, BA2 = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

for i in range(num_epochs):
    loss = model.fit(train_dataset)
    y_pred = model.predict(valid_dataset)
    y_pre = y_pred[:, 1]
    auc_roc_score_test = roc_auc_score(np.array(valid_true).squeeze(), y_pre)
    print("AUC:",auc_roc_score_test)
    if auc_roc_score_test > best_roc_auc_score:
        best_roc_auc_score = auc_roc_score_test
        tp, tn, fn, fp, se, sp, mcc, q, auc_roc_score, F1, BA = score(train_dataset)
        tp1, tn1, fn1, fp1, se1, sp1, mcc1, q1, auc_roc_score1, F1_1, BA1 = tp, tn, fn, fp, se, sp, mcc, q, auc_roc_score, F1, BA
        tp, tn, fn, fp, se, sp, mcc, q, auc_roc_score, F1, BA = score(test_dataset)
        tp2, tn2, fn2, fp2, se2, sp2, mcc2, q2, auc_roc_score2, F1_2, BA2 = tp, tn, fn, fp, se, sp, mcc, q, auc_roc_score, F1, BA

print('train_score:')
print('TP:', tp1, 'FN:', fn1, 'TN:', tn1, 'FP:', fp1)
print('SE:', se1, 'SP:', sp1)
print('MCC:', mcc1, 'Q:', q1)
print('auc_roc:', auc_roc_score1)
print('F1:', F1_1)


print('test_score:')
print('TP:', tp2, 'FN:', fn2, 'TN:', tn2, 'FP:', fp2)
print('SE:', se2, 'SP:', sp2)
print('MCC:', mcc2, 'Q:', q2)
print('auc_roc:', auc_roc_score2)
print('F1:', F1_2)