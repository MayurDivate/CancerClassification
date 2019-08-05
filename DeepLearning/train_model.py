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

    def __init__(self, train_tsv, test_tsv, val_tsv, nfeatures, nclasses, mname, outdir="./"):
        self.train_tsv = train_tsv
        self.test_tsv = test_tsv
        self.val_tsv = val_tsv
        self.nfeatures = nfeatures
        self.nclasses = nclasses
        self.mname = mname
        self.outdir = outdir

    def run_cnn_model(self, e=20, test=False):
        self.predict_txt = self.mname + "_1D_CNN_predict.txt"

        # get the training data
        Ptrain = Preprocessor(input_files=[self.train_tsv], nfeatures=self.nfeatures)
        Pval = Preprocessor(input_files=[self.val_tsv], nfeatures=self.nfeatures)

        train_exp, train_lab = Ptrain.get_cnn_data()
        val_exp, val_lab = Pval.get_cnn_data()

        print(val_exp.shape, train_exp.shape)

        # initialize model
        cnn_model = DLmodel(self.nfeatures, self.nclasses).get_1D_cnn_model()

        # train the model
        train_results = cnn_model.fit(train_exp, train_lab, epochs=e, validation_data=(val_exp, val_lab))
        MyPlotter = Plotter(outimg=self.mname + " 1D CNN", outdir=self.outdir)
        MyPlotter.plot_accuracy_and_loss(train_results)

        if test:
            # get the test data
            Ptest = Preprocessor([self.test_tsv], self.nfeatures)
            test_exp, test_lab = Ptest.get_cnn_data()
            ypred = cnn_model.predict(test_exp)
            self.print_ypred_test_labels(ypred, test_lab)
            print("Results dir: ", self.outdir)

        return cnn_model, self.get_model_accuracy_and_loss(train_results)

    def run_mlp_model(self, e=20, test=False):
        self.predict_txt = self.mname + "_MLP_predict.txt"

        # get the training data
        Ptrain = Preprocessor(input_files=[self.train_tsv], nfeatures=self.nfeatures)
        Pval = Preprocessor(input_files=[self.val_tsv], nfeatures=self.nfeatures)

        train_exp, train_lab = Ptrain.get_mlp_data()
        val_exp, val_lab = Pval.get_mlp_data()

        # initialize model
        mlp_model = DLmodel(self.nfeatures, self.nclasses).get_mlp_model()

        # train the model
        train_results = mlp_model.fit(train_exp, train_lab, epochs=e, validation_data=(val_exp, val_lab))

        MyPlotter = Plotter(outimg=self.mname + " MLP", outdir=self.outdir)
        MyPlotter.plot_accuracy_and_loss(train_results)

        if test:
            # get the test data
            Ptest = Preprocessor([self.test_tsv], self.nfeatures)
            test_exp, test_lab = Ptest.get_mlp_data()
            ypred = mlp_model.predict(test_exp)
            self.print_ypred_test_labels(ypred, test_lab)
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






