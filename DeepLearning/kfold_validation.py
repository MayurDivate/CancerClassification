from .data_preprocess import Preprocessor
from .models import DLmodel
from .plots import Plotter


# 5-fold cross validation

class K5FoldCrossValidation():

    def __init__(self, k1, k2, k3, k4, k5, mname, nfeatures, nclasses, outdir):
        self.k = list([k1, k2, k3, k4, k5])
        self.nfeatures = nfeatures
        self.mname = mname
        self.outdir = outdir
        self.nclasses = nclasses

    # entry method to run kfold cross validation
    def run_kfold_cross_validation(self, epochs=20, cnn=True):

        res_kfold = list()  # k1-5 results
        for i in range(len(self.k)):
            k_training_sets = list([])  # training sets
            out_file = self.mname + "_K" + str(i + 1)  # plot file basename

            for j in range(len(self.k)):
                k_val = self.k[i]  # validation set
                if i != j:
                    k_training_sets.append(self.k[j])

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

    def run_cnn_model(self, k_train, k_val, e):
        # print("Training sets:", k_train)
        # print("Validation set:", k_val)

        # get data from the file and pre-process it
        Ptrain = Preprocessor(input_files=k_train, nfeatures=self.nfeatures)
        train_exp, train_lab = Ptrain.get_cnn_data()

        Pval = Preprocessor(input_files=[k_val], nfeatures=self.nfeatures)
        val_exp, val_lab = Pval.get_cnn_data()

        # initialize the model
        model = DLmodel(self.nfeatures, self.nclasses).get_1D_cnn_model()
        print("---- DONE ----")
        return model.fit(train_exp, train_lab, epochs=e, validation_data=(val_exp, val_lab))


    def run_mlp_model(self, k_train, k_val, e):
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