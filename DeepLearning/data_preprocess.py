import numpy as np
import pandas as pd


# data preprocesser
class Preprocessor:

    def __init__(self, input_files, nfeatures):
        self.input_files = input_files
        self.nfeatures = nfeatures

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
                df = pd.read_csv(f, sep='\t', skiprows=1)
                exp_batch = df.iloc[:, :-1]
                exp = exp.append(exp_batch, sort=True)

                lab_batch = list(df.iloc[:, -1])
                lab = lab.append(lab_batch, sort=True)

        exp = np.array(exp, np.float32)
        lab = np.array(lab)
        lab = self.get_one_hot_encoded_labels(lab)
        return exp, lab

    def get_one_hot_encoded_labels(self, labels):

        label_list = list()

        for label in labels:
            if label == 'ALL':
                label_list.append([1, 0, 0, 0, 0])
            elif label == 'AML':
                label_list.append([0, 1, 0, 0, 0])
            elif label == 'Bcell':
                label_list.append([0, 0, 1, 0, 0])
            elif label == 'Breast':
                label_list.append([0, 0, 0, 1, 0])
            elif label == 'skin':
                label_list.append([0, 0, 0, 0, 1])

        return np.array(label_list, np.float32)
