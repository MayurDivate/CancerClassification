import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


class ClassificationReport:

    def __init__(self, model_res_file, ypred_col, lfile, real_col, outfile):
        self.model_res_file = model_res_file
        self.ypred_col = ypred_col
        self.real_col = real_col

        self.lfile = lfile
        self.labels = [label.rstrip() for label in open(self.lfile, 'r')]
        self.outfile = outfile

    def parse_results(self):
        df = pd.read_csv(self.model_res_file, sep='\t', header=None).dropna(axis=1)
        ypred = np.array(df.iloc[:, self.ypred_col])
        yreal = np.array(df.iloc[:, self.real_col])

        res = confusion_matrix(yreal, ypred)
        acc = accuracy_score(yreal, ypred)
        print("Accuracy:", acc)
        report = classification_report(yreal, ypred)
        print("Classification: ",report)
        out = os.path.join(self.outfile)

        with open(out, 'w') as f:
            f.write(report)

        print('Done')


# Example

# model_1 = '/Users/n10337547/Projects/model_2_pred.txt'
# out = '/Users/n10337547/Projects/3_Symposium/3_Results'
# lfile = '/Users/n10337547/Projects/1_CUP/2_TCGA/0_Data/TCGA_classes.txt'
# model_1_conf = ClassificationReport(model_1, 78, 79, lfile, out)
# model_1_conf.parse_results()
