import numpy as np
from DeepLearning.data_preprocess import 
from tensorflow.python.keras import models
import pandas as pd

class PretrainedModel:
    def __init__(self, model_file, data_file, nf, outfile):
        self.model_file = model_file
        self.data_file = data_file
        self.nfeatures = nf
        self.outfile = outfile

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


