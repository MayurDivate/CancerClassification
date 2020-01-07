import numpy as np
import os
from tensorflow.python.keras import models
import pandas as pd

class PretrainedModel:
    def __init__(self):
        self.model_file = '/Users/n10337547/Projects/3_Symposium/3_Results/without_metastatic/WM_FPKM_5_1D_CNN.h5'
        self.data_file = '/Users/n10337547/Projects/1_CUP/3_dbGaP/3_FPKM_mats/dbGaP_FPKM_5_master.tsv'
        self.nfeatures = 13488
        self.outfile = '/Users/n10337547/Projects/1_CUP/3_dbGaP/3_FPKM_mats/dbGaP_FPKM_5_pred.tsv'

    def load_and_run_model(self):
        model = models.load_model(self.model_file)
        exp = self.get_cnn_data()
        res = model.predict(exp)
        self.print_ypred_test_labels(res)

    def get_cnn_data(self):
        exp = self.get_mlp_data()
        exp = exp.reshape(exp.shape[0], 1, self.nfeatures)
        return exp

    def get_mlp_data(self):
        df = pd.read_csv(self.data_file, sep="\t", index_col=0)
        df = df.drop('label', axis=1)
        df = df.to_numpy('float32')
        return df

    def print_ypred_test_labels(self, ypred):
        ypred_argmax = np.argmax(ypred, 1)
        ylen = ypred.shape[0]
        yclasses = ypred.shape[1]

        with open(self.outfile, 'w+') as out:
            for rec in range(ylen):
                for i in range(yclasses):
                    out.write(str(ypred[rec, i]))
                    out.write('\t')

                out.write(str(ypred_argmax[rec]))
                out.write('\n')


