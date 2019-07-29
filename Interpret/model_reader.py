import os

import numpy as np
import pandas as pd
from tensorflow.python import keras as ks

from Interpret.feature_scores import ImportanceScoreCalculator, CnnFilterScoreCalculator


class ModelReader:
    def print_info(self):
        print("Model: ", self.model_file)
        print("Layers: ", self.layer_count)

    def get_layers_count(self):
        return int(len(self.get_parameters()) / 2)

    def get_parameters(self):
        return ks.models.load_model(self.model_file).get_weights()

    def get_parameter_lists(self):
        bias_list = list()
        weight_list = list()

        parameters = self.get_parameters()

        ibias = self.layer_count * 2 - 1
        iweight = ibias - 1
        layers = reversed(list(range(self.layer_count)))
        self.last_layer_imp_scores = np.ones(len(parameters[ibias]))

        for loop in layers:
            bias_list.append(parameters[ibias])
            weight_list.append(parameters[iweight])
            ibias -= 2
            iweight -= 2

        return weight_list, bias_list


class CNNmodleReader(ModelReader):

    def __init__(self, model_file):
        self.model_file = model_file

        self.layer_count = self.get_layers_count()
        self.weights, self.biases = self.get_parameter_lists()

    def get_feature_importance_score(self):

        last_layer_imp_scores = np.ones(len(self.biases[0]))
        print('\nCalculating the importance score...')

        for i in range(self.layer_count - 1):
            print(">>> layer:", i + 1, " Shape :", self.weights[i].shape, "bias: ", self.biases[i].shape)
            ISC = ImportanceScoreCalculator(self.weights[i], self.biases[i], last_layer_imp_scores)
            last_layer_imp_scores = ISC.calculate_imp_score()

        # scores for filters
        last_layer = self.layer_count - 1
        print("\nCalculating scores for filters...")
        CFS = CnnFilterScoreCalculator(self.weights[last_layer], self.biases[last_layer], last_layer_imp_scores)
        last_layer_imp_scores = CFS.calculate_imp_score()
        print('\nFinished!\n')
        return last_layer_imp_scores

    def write_scores_csv(self, feature_scores, output_file):

        df = pd.DataFrame()
        sumx = list(feature_scores.sum(axis=2))
        sumx = sumx[0]

        for scores in feature_scores[0]:
            dfX = pd.DataFrame(np.round(scores,4))
            dfX = dfX.T
            df = df.append(dfX, ignore_index=True)

        df['sum'] = sumx
        df.to_csv(output_file, sep='\t')

        print('Scores have been written in the file: ', os.path.basename(output_file))

class MLPmodelReader(ModelReader):
    def __init__(self, model_file):
        self.model_file = model_file

        self.layer_count = self.get_layers_count()
        self.weights, self.biases = self.get_parameter_lists()

    def get_feature_importance_score(self):
        last_layer_imp_scores = np.ones(len(self.biases[0]))
        print('\nCalculating the importance score...')
        for i in range(self.layer_count):
            print(">>> layer:", i)
            ISC = ImportanceScoreCalculator(self.weights[i], self.biases[i], last_layer_imp_scores)
            last_layer_imp_scores = ISC.calculate_imp_score()

        print('Finished!\n')
        return last_layer_imp_scores

    def write_scores_csv(self, feature_scores, output_file):

        with open(output_file, 'w') as f:
            f.write('FeatureIndex\tScore\n')
            for i in range(len(feature_scores)):
                f.write('{}\t{}\n'.format(i, round(feature_scores[i], 4)))
                f.flush()
        f.close()

        print('Scores have been written in the file: ', os.path.basename(output_file))
