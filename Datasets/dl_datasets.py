import random
import os

from Matrix.filters import RowFilter


class Samples:

    def __init__(self, sample_list_file, ftrain=0.7, fval=0.1, ftest=0.2, limit=None):
        self.sample_list_file = sample_list_file
        self.sample_list = [file.rstrip() for file in open(self.sample_list_file, 'r')]
        self.limit = limit
        self.ftrain = ftrain
        self.ftest = ftest
        self.fval = fval

    def get_train_test_val_sets(self):
        self.sample_list = self.get_randomized_list()
        ntrain, nval, ntest = self.get_dataset_sizes()

        train_samples = self.sample_list[:ntrain]
        val_samples = self.sample_list[ntrain:(ntrain + nval)]
        test_samples = self.sample_list[(ntrain + nval):]

        return train_samples, val_samples, test_samples

    def get_dataset_sizes(self):
        nf = len(self.sample_list)

        train_val_test = [round(0.7 * nf), round(0.1 * nf), round(0.2 * nf)]

        total = sum(train_val_test)

        if nf < total:
            index = train_val_test.index(max(train_val_test))
            train_val_test[index] = train_val_test[index] - (total - nf)

        return train_val_test[0], train_val_test[1], train_val_test[2]

    def get_randomized_list(self):

        for i in [1, 2, 3, 4, 5]:
            random.shuffle(self.sample_list)

        return self.sample_list


class TrainingTestingData:

    def __init__(self, master_mat_file, train_samples, val_samples, test_samples, outdir):
        self.master_mat_file = master_mat_file
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.test_samples = test_samples
        self.outdir = outdir

    def create_dl_datasets(self):
        print('Training set...')
        train_set = RowFilter(self.master_mat_file, self.train_samples, os.path.join(self.outdir, 'train_mat.tsv'))
        train_set.apply_filter()

        print('Test set...')
        test_set = RowFilter(self.master_mat_file, self.test_samples, os.path.join(self.outdir, 'test_mat.tsv'))
        test_set.apply_filter()

        print('Validation set...')
        val_set = RowFilter(self.master_mat_file, self.val_samples, os.path.join(self.outdir, 'val_mat.tsv'))
        val_set.apply_filter()

        print('Done')
