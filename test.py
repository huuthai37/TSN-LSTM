import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, precision_recall_fscore_support, accuracy_score, classification_report
import pandas as pd
import seaborn as sn
import pickle

out_file = 'results/incept299_spatial/incept229_spatial_lstm256_45e_cr1.pickle'
with open(out_file,'rb') as f1:
    data = pickle.load(f1)

out_file = 'results/temporal/incept229_temporal_lstm256_35e_cr1.pickle'
with open(out_file,'rb') as f1:
    data2 = pickle.load(f1)

out_file = 'results/t2/incept229_temporal2_lstm256_32e_cr1.pickle'
with open(out_file,'rb') as f1:
    data22 = pickle.load(f1)

out_file = 'results/last/incept229_twostream1_lstm256_18e_cr1.pickle'
with open(out_file,'rb') as f1:
    data3 = pickle.load(f1)

out_file = 'results/last/incept229_twostream2_lstm256_109e_cr1.pickle'
with open(out_file,'rb') as f1:
    data32 = pickle.load(f1)

y_pred = data[0].argmax(axis=-1)
y_pred2 = data2[0].argmax(axis=-1)
y_pred22 = data22[0].argmax(axis=-1)
y_pred3 = data3[0].argmax(axis=-1)
y_pred32 = data32[0].argmax(axis=-1)
class_file = 'data/ucf101-classInd.txt'
classInd=[]
with open(class_file) as f0:
    for line in f0:
        class_name = line.rstrip()
        if class_name:
            classInd.append(class_name)

print classification_report(data3[1], y_pred3, digits=6)
print precision_recall_fscore_support(data3[1], y_pred3, average='macro')
print accuracy_score(data3[1], y_pred3)