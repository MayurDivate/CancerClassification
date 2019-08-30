import os
import random

from Matrix.filters import RowFilter


class Samples:

    def __init__(self, sample_list_file, ftrain=0.7, fval=0.1, ftest=0.2, limit=None):
        self.sample_list_file = sample_list_file

        if not isinstance(sample_list_file, list):
            self.sample_list = [file.rstrip() for file in open(self.sample_list_file, 'r')]
        else:
            self.sample_list = sample_list_file

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

class KfoldData:

    def __init__(self, master_mat_file, samples, outfile):
        self.master_mat_file = master_mat_file
        self.samples= samples
        self.outfile = outfile

    def create_dl_datasets(self):
        print('kfold data')
        train_set = RowFilter(self.master_mat_file, self.samples, self.outfile)
        train_set.apply_filter()

        print('Done')



class Kfold_sets:

    def __init__(self, k, sample_list, outdir, suffix=''):
        self.k = k
        self.suffix = suffix
        self.samples = [file.rstrip() for file in open(sample_list, 'r')]
        self.set_sizes = self.get_set_sizes()
        self.outdir = outdir

    def distribute_files_randomly(self):

        print()
        print(f'distributing files in {self.k} sets')
        print()

        for i in range(self.k):
            random.shuffle(self.samples)

        start = 0
        end = 0
        k_dict = dict()

        for k in range(self.k - 1):
            end = end + self.set_sizes[k]
            key = "k" + str(k + 1)
            k_dict[key] = self.samples[start:end]
            start = end

        k = "k" + str(self.k)
        k_dict[k] = self.samples[start:]

        self.write_k_list(k_dict)

        print("---------- DONE ----------")

    def get_set_sizes(self):
        ksize = round(1 / self.k, 2)
        nf = len(self.samples)
        size = round(ksize * nf)
        set_sizes = []

        for i in range(self.k):
            nf = nf - size
            set_sizes.append(size)

        set_sizes.append(nf)
        # print(set_sizes)
        return set_sizes

    def write_k_list(self, k_dict):
        for key in k_dict:

            outfile = os.path.join(self.outdir, key + '_' + self.suffix+'.txt')
            print(key, outfile)

            with open(outfile, 'w') as f:
                for kfile in k_dict[key]:
                    f.write(kfile + '\n')

