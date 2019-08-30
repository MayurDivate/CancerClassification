import os

from Datasets.dl_datasets import KfoldData
from DeepLearning.data_preprocess import Preprocessor
from DeepLearning.models import DLmodel
from .plots import Plotter


# K fold cross validation

class KFoldCrossValidation():

    def __init__(self, ks_list, master_matrix, mname, nfeatures, nclasses, outdir):
        self.k_list = ks_list
        self.master_matrix = master_matrix
        self.nfeatures = nfeatures
        self.mname = mname
        self.outdir = outdir
        self.nclasses = nclasses

    # entry method to run kfold cross validation
    def run_kfold_cross_validation(self, epochs=20, cnn=True):
        print("run_kfold_cross_validation()")

        res_kfold = list()  # k1-5 results

        for i in range(len(self.k_list)):
            k_training_sets = list([])  # training sets
            out_file = str.replace(self.mname + " K" + str(i + 1), " ", "_")  # plot file basename
            print(out_file)

            for j in range(len(self.k_list)):
                k_val = self.k_list[i]  # validation set
                if i != j:
                    k_training_sets.append(self.k_list[j])

            # run model
            print(">>> kth fold = " + str(i + 1))
            if cnn:
                train_results = self.run_cnn_model(k_training_sets, k_val, epochs)
                res_kfold.append(train_results)
            else:
                train_results = self.run_mlp_model(k_training_sets, k_val, epochs)
                res_kfold.append(train_results)

            """
            # print kth fold results
            for key in train_results.history.keys():
                print(key, " = ", train_results.history[key])
            """
            # plot kth fold results
            MyPlotter = Plotter(outimg=out_file, outdir=self.outdir)
            MyPlotter.plot_accuracy_and_loss(results=train_results)

        # calculate the average score
        self.get_avg_score(res_kfold)
        print("---- Finished ----")

    def get_avg_score(self, res_kfold):

        k = 0
        acc = 0
        val_acc = 0

        for i in res_kfold:
            k = k + 1
            acc = acc + i.history['accuracy'][-1]
            val_acc = val_acc + i.history['val_accuracy'][-1]

        acc = round(acc / k, 4)
        val_acc = round(val_acc / k, 4)

        print("k fold Accuracy: ", acc)
        print("k fold Val accuracy:", val_acc)

    def get_samples_from_lists(self, k_lists):
        samples = list()
        for f in k_lists:
            samples = samples + [sample.rstrip() for sample in open(f, 'r')]

        return samples

    def run_cnn_model(self, k_train, k_val, e):
        print('CNN model')

        k_train_samples = self.get_samples_from_lists(k_train)
        k_val_samples = self.get_samples_from_lists([k_val])

        k_train_out = os.path.join(self.outdir, 'k_train.tsv')
        k_val_out = os.path.join(self.outdir, 'k_val.tsv')

        KfoldData(self.master_matrix, k_train_samples, k_train_out).create_dl_datasets()
        KfoldData(self.master_matrix, k_val_samples, k_val_out).create_dl_datasets()

        # get data from the file and pre-process it
        Ptrain = Preprocessor(input_files=[k_train_out], nfeatures=self.nfeatures)
        train_exp, train_lab = Ptrain.get_cnn_data()

        Pval = Preprocessor(input_files=[k_val_out], nfeatures=self.nfeatures)
        val_exp, val_lab = Pval.get_cnn_data()

        # initialize the model
        model = DLmodel(self.nfeatures, self.nclasses).get_1D_cnn_model()
        print("---- DONE ----")
        return model.fit(train_exp, train_lab, epochs=e, validation_data=(val_exp, val_lab))

    def run_mlp_model(self, k_train, k_val, e):

        pass
        """
        # print("Training sets:", k_train)
        # print("Validation set:", k_val)

        # get data from the file and pre-process it
        Ptrain = Preprocessor(input_files=k_train, nfeatures=self.nfeatures)
        train_exp, train_lab = Ptrain.get_mlp_data()

        Pval = Preprocessor(input_files=[k_val], nfeatures=self.nfeatures)
        val_exp, val_lab = Pval.get_mlp_data()

        # initialize the model
        model = DLmodel(self.nfeatures, self.nclasses).get_mlp_model()
        print("---- DONE ----")
        return model.fit(train_exp, train_lab, epochs=e, validation_data=(val_exp, val_lab))
        """
