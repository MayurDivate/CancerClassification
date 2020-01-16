import numpy as np
from DeepLearning.data_preprocess import Preprocessor
from tensorflow.python.keras import models
import pandas as pd

class PretrainedModel:
    def __init__(self, model_file, data_file, outfile, label_file):
        self.model_file = model_file
        self.data_file = data_file
        self.outfile = outfile
        self.label_file = label_file

    def load_and_run_model(self):
        model = models.load_model(self.model_file)
        dataX = Preprocessor(self.data_file, self.label_file).get_test_data()
        res = model.predict(dataX)
        self.print_ypred_test_labels(res)

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


