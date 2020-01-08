import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

class Preprocessor:
    def __init__(self, inputfile, label_file):
        self.input_files = inputfile
        self.labels_file = label_file

    def split_the_data(self):
        data = pd.read_csv(self.input_files, sep='\t', index_col=0)
        dataY = data.iloc[:, -1] # last column contains labels
        data = data.iloc[:, :-1] # droping last columns
        trainX, testX, trainY, testY = train_test_split(data, dataY, test_size=0.2, stratify=dataY)
        trainY = self.get_one_encoded_labels(trainY)
        testY = self.get_one_encoded_labels(testY)

        return trainX, testX, trainY, testY

    def get_one_encoded_labels(self, y):
        y = y.to_numpy().reshape(-1, 1)
        labels = np.array([label.strip() for label in open(self.labels_file)]).reshape(-1,1)
        ohe = OneHotEncoder()
        ohe.fit(labels)
        return ohe.transform(y).toarray()

    def get_mlp_data(self):
        trainX, testX, trainY, testY = self.split_the_data()
        trainX = trainX.to_numpy('float64')
        testX = testX.to_numpy('float64')

        return trainX, testX, trainY, testY

    def get_cnn_data(self):
        trainX, testX, trainY, testY =  self.split_the_data()
        trainX = self.logtranformthe_data(self.reshape_data(trainX))
        testX  = self.logtranformthe_data(self.reshape_data(testX))

        return trainX, testX, trainY, testY

    def reshape_data(self,x):
        x = x.to_numpy('float64')
        return x.reshape(x.shape[0], 1, x.shape[1])

    def logtranformthe_data(self, x, add_to_zeros = 0.001):
        x = np.log(x+ add_to_zeros)
        return x

# old data preprocesser
class Old_Preprocessor:


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

