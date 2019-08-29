import numpy as np
import pandas as pd


# data preprocesser
class Preprocessor:

    def __init__(self, input_files, nfeatures, labels_file):
        self.input_files = input_files
        self.nfeatures = nfeatures
        self.labels_file = labels_file
        self.ohe_dict = self.get_one_hot_encoded_dict()

    def get_one_hot_encoded_dict(self):
        labels = [label.rstrip() for label in open(self.labels_file, 'r')]
        nclasses = len(labels)
        ohe_dict = {}

        for i, label in enumerate(labels):
            ohe = [0] * nclasses
            ohe[i] = 1
            ohe_dict[label] = ohe

        return ohe_dict


    def get_cnn_data(self):
        exp, lab = self.get_mlp_data()

        # 1 column matrix shape
        # exp = exp.reshape(exp.shape[0], self.nfeatures, 1)

        # 1 row matrix shape
        exp = exp.reshape(exp.shape[0], 1, self.nfeatures)

        return exp, lab

    def get_mlp_data(self):
        exp = pd.DataFrame()
        lab = pd.DataFrame()

        for k in self.input_files:

            with open(k, 'r') as f:
                df = pd.read_csv(f, sep='\t', index_col=0)
                exp_batch = df.iloc[:, :-1]
                exp = exp.append(exp_batch)

                lab_batch = list(df.iloc[:, -1])
                lab = lab.append(lab_batch)

        lab = lab.to_numpy()
        exp = exp.to_numpy('float32')

        lab = self.get_one_hot_encoded_labels(lab)
        return exp, lab

    def get_one_hot_encoded_labels(self, labels):

        label_list = list()

        for label in labels:
            label_list.append(self.ohe_dict[label[0]])

        return np.array(label_list, np.float32)
