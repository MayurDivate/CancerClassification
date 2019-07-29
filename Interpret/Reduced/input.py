import os

import pandas as pd

from Data.TrainValTestInput import TrainTestValInput


class ReducedInput:

    def __init__(self, raw_data_dir, geneinfo, outputdir,
                 labels='../1_DL_data/Labels/BloodCancerLabels.txt', genetype='lincRNA'):
        self.raw_data_dir = raw_data_dir
        self.geneinfo = geneinfo
        self.outputdir = outputdir
        self.genetype = genetype
        self.labels = labels

        self.train_dir = os.path.join(self.raw_data_dir, 'Train')
        self.test_dir = os.path.join(self.raw_data_dir, 'Test')
        self.val_dir = os.path.join(self.raw_data_dir, 'Val')

    def create_reduced_input(self):
        train_val_test = TrainTestValInput(self.train_dir, self.test_dir, self.val_dir, self.labels,
                                           self.geneinfo, self.genetype, self.outputdir)
        train_val_test.create_train_val_test_tsv()

class InputTsvReducer:

    def __init__(self, input_tsv, reduce_by):
        self.input_tsv = input_tsv

    def reduce_tsv(self, reduce_by):
        df = pd.read_csv(self.input_tsv, sep='\t')
        print(df.head)

class HeatmapInput:

    def __init__(self, input_tsv_dir, outfile=None):
        self.input_tsv_dir = input_tsv_dir
        self.train = os.path.join(self.input_tsv_dir, 'Train.tsv')
        self.test = os.path.join(self.input_tsv_dir, 'Test.tsv')
        self.val = os.path.join(self.input_tsv_dir, 'Val.tsv')

        if outfile == None:
            self.outfile = os.path.join(input_tsv_dir, 'Heatmap.tsv')

    def create_heatmp_input_tsv(self):

        df = pd.read_csv(self.train, sep='\t', skiprows=1)
        df = df.append(pd.read_csv(self.test, sep='\t', skiprows=1))
        df = df.append(pd.read_csv(self.val, sep='\t', skiprows=1))
        df = df.T

        df.columns = self.get_pseudo_sample_names(df)
        print(df.head())

        df.to_csv(self.outfile, sep='\t')
        print("output file: ", self.outfile)

    def get_pseudo_sample_names(self, df):

        colname = df.loc['label']
        sample = pd.DataFrame()

        for label in colname.unique():
            sample_number = 1
            original_index = 0

            for sample_prefix in colname:

                if sample_prefix == label:
                    sample_number += 1
                    sample_name = sample_prefix + "_" + str(sample_number)
                    sample = sample.append(pd.DataFrame([sample_name],index=[original_index]))

                original_index += 1
        sample = sample.sort_index()
        return list(sample[0])
