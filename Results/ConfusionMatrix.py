import numpy as np
import pandas as pd
import os

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


class ConfusionMatrix:

    def __init__(self, model_res_file, ypred_col, real_col, outdir):
        self.model_res_file = model_res_file
        self.ypred_col = ypred_col
        self.real_col = real_col
        lfile = '/Users/n10337547/Projects/1_CUP/2_TCGA/0_Data/TCGA_classes.txt'
        self.labels = [label.rstrip() for label in open(lfile,'r')]
        self.outdir = outdir

    def parse_results(self):
        df = pd.read_csv(self.model_res_file, sep='\t', header=None).dropna(axis=1)
        ypred = np.array(df.iloc[:, self.ypred_col])
        yreal = np.array(df.iloc[:, self.real_col])


        res =  confusion_matrix(yreal, ypred)
        print(res)
        acc  = accuracy_score(yreal, ypred)
        print(acc)
        report = classification_report(yreal, ypred)
        print(report)
        out = os.path.join(self.outdir, 'classification_report.txt')


        with open(out, 'w') as f:
            f.write(report)

        print('Done')


model_1 = '/Users/n10337547/Projects/model_2_pred.txt'
out = '/Users/n10337547/Projects/3_Symposium/3_Results'

model_1_conf = ConfusionMatrix(model_1, 78, 79, out)
model_1_conf.parse_results()
