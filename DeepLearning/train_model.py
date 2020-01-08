import os

import numpy as np
import pandas as pd

from .data_preprocess import Preprocessor
from .models import DLmodel
from .plots import Plotter


# Train Test Validate
class ModelTrainer:
    """
    This class object can be initialized using three files i.e. trian, test and validataion CSV
    and the number of features (int) per sample
    Call train_model method to train the model using the data in the train.tsv and val data in the val.tsv
    then this method itself calls plot_accuracy_and_loss method of Plotter class to create plots
    after that test dataset will be used to predict results. finally prediction results will be
    laid out in the out.txt
    """

    def __init__(self, inputdata, labels_file, mname, outdir="./"):
        self.inputdata = inputdata
        self.labels_file = labels_file
        self.mname = mname
        self.outdir = outdir
        self.nclasses = self.get_number_of_classes()

    def get_number_of_classes(self):
        return len([c for c in open(self.labels_file, 'r')])

    def run_cnn_model(self, e=20, test=False):
        self.predict_txt = self.mname + "_1D_CNN_predict.txt"
        self.result_log = os.path.join(self.outdir, self.mname + "_1D_CNN.log")

        # get the train and test data
        trainX, testX, trainY, testY = Preprocessor(self.inputdata, self.labels_file).get_cnn_data()
        self.nfeatures = trainX.shape[-1]

        # initialize model
        cnn_model = DLmodel(self.nfeatures, self.nclasses).get_1D_cnn_model()

        # train the model
        train_results = cnn_model.fit(trainX, trainY, epochs=e, validation_split=0.125)

        pd.DataFrame(train_results.history).to_csv(self.result_log, sep='\t')

        MyPlotter = Plotter(outimg=self.mname + " 1D CNN", outdir=self.outdir)
        MyPlotter.plot_accuracy_and_loss(train_results)


        if test:
            # get the test data
            ypred = cnn_model.predict(testX)

            self.print_ypred_test_labels(ypred, testY)
            print("Results dir: ", self.outdir)

        return cnn_model, self.get_model_accuracy_and_loss(train_results)

    def run_mlp_model(self, e=20, test=False):
        self.predict_txt = self.mname + "_MLP_predict.txt"
        self.result_log = os.path.join(self.outdir,self.mname + "_MLP.log")

        # get the training data
        trainX, testX, trainY, testY = Preprocessor(self.inputdata, self.labels_file).get_mlp_data()

        # initialize model
        mlp_model = DLmodel(self.nfeatures, self.nclasses).get_mlp_model()

        # train the model
        train_results = mlp_model.fit(trainX, trainY, epochs=e, validation_split=0.125)

        pd.DataFrame(train_results.history).to_csv(self.result_log, sep='\t')

        MyPlotter = Plotter(outimg=self.mname + " MLP", outdir=self.outdir)
        MyPlotter.plot_accuracy_and_loss(train_results)

        if test:
            # get the test data
            ypred = mlp_model.predict(testX)
            self.print_ypred_test_labels(ypred, testY)
            print("Results dir: ", self.outdir)

        return mlp_model, self.get_model_accuracy_and_loss(train_results)

    def get_model_accuracy_and_loss(self, fit_results):
        res = {'accuracy': fit_results.history['accuracy'][-1],
               'val_accuracy': fit_results.history['val_accuracy'][-1],
               'loss': fit_results.history['loss'][-1],
               'val_loss': fit_results.history['val_loss'][-1]}

        return res

    def print_ypred_test_labels(self, ypred, labels):
        ypred_argmax = np.argmax(ypred, 1)
        lab_argmax = np.argmax(labels, 1)
        outfile = os.path.join(self.outdir, self.predict_txt)

        ylen = ypred.shape[0]
        yclasses = ypred.shape[1]

        with open(outfile, 'w+') as out:
            for rec in range(ylen):
                for i in range(yclasses):
                    out.write(str(ypred[rec, i]))
                    out.write('\t')
                    out.write(str(labels[rec, i]))
                    out.write('\t')

                out.write('\t')
                out.write(str(ypred_argmax[rec]))
                out.write('\t')
                out.write(str(lab_argmax[rec]))
                out.write('\t')
                out.write(str(ypred_argmax[rec] == lab_argmax[rec]))
                out.write('\n')

    def print_model_training_progress(self, res):

        df = pd.DataFrame(res)
        df.to_csv('training_progress.tsv', sep='\t')
