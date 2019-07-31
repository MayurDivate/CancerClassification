import random


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
