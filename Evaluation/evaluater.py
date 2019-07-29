import numpy as np
import pandas as pd
import tensorflow as tf

from DeepLearning.data_preprocess import Preprocessor


class ModelEvaluater():

    def __init__(self, model_file, input_tsv, nfeatures, outfile):
        self.model_file = model_file
        self.input_tsv = input_tsv
        self.outfile = outfile
        self.nfeatures = nfeatures

    def run_evaluation(self):

        TestData = Preprocessor(input_files=[self.input_tsv], nfeatures=self.nfeatures)
        x, labels = TestData.get_cnn_data()

        model = tf.keras.models.load_model(self.model_file)
        res = model.predict(x)

        self.print_ypred_test_labels(res, labels)

    def print_ypred_test_labels(self, ypred, labels):
        ypred_argmax = np.argmax(ypred, 1)
        lab_argmax = np.argmax(labels, 1)

        dictX = [{'yred': list(ypred[i]),
                  'labels': list(labels[i]),
                  'ypred_argmax': ypred_argmax[i],
                  'label_argmax': lab_argmax[i],
                  'is true': (ypred_argmax[i] == lab_argmax[i])} for i in range(len(lab_argmax))]

        df = pd.DataFrame(dictX)

        df.to_csv(self.outfile, sep='\t')
