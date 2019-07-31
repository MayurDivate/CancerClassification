import os
import random

from Data.Sets.utils import copy_files_to_dir, create_output_dir, cleandir


class TrainValTestSets:
    """
    70 10 20 distribution of files
    """

    def __init__(self, source_dir, target_dir):
        self.source_dir = source_dir
        self.target_dir = target_dir

        # output dirs
        self.train_dir = os.path.join(target_dir, 'Train')
        self.val_dir = os.path.join(target_dir, 'Val')
        self.test_dir = os.path.join(target_dir, 'Test')

        # files in the source dir
        self.files = [f for f in os.listdir(source_dir) if not f.startswith('.')]

        # 70 10 20 counts for Train Test val
        self.ntrain, self.nval, self.ntest = self.get_set_count()

    def get_set_count(self):
        nf = len(self.files)

        train_val_test = [round(0.7 * nf), round(0.1 * nf), round(0.2 * nf)]

        total = sum(train_val_test)

        if nf < total:
            index = train_val_test.index(max(train_val_test))
            train_val_test[index] = train_val_test[index] - (total - nf)

        return train_val_test[0], train_val_test[1], train_val_test[2]

    def distribute_files_randomly(self):
        print()
        print("distributing files in Train, Val and Test dataset")
        print(f'Source dir: {os.path.basename(self.source_dir)};'
              f'Train: {self.ntrain}, Val: {self.nval}, Test:{self.ntest}')
        print()
        # create output dirs
        create_output_dir(self.train_dir)
        create_output_dir(self.test_dir)
        create_output_dir(self.val_dir)

        # shuffle the files
        for i in [1, 2, 3]:
            random.shuffle(self.files)

        # divide files in train val and test sets
        train_files = self.files[:self.ntrain]
        val_files = self.files[self.ntrain:(self.ntrain + self.nval)]
        test_files = self.files[(self.ntrain + self.nval):]

        # copy files to target folders
        copy_files_to_dir(train_files, self.source_dir, self.train_dir)
        copy_files_to_dir(val_files, self.source_dir, self.val_dir)
        copy_files_to_dir(test_files, self.source_dir, self.test_dir)

        print("---------- DONE ----------")


class Kfold_sets:
    """
    5 fold cross validation file distribution
    """

    def __init__(self, source_dir, target_dir, kfold):
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.kfold = kfold

        # get files in source dir
        self.files = [f for f in os.listdir(source_dir) if not f.startswith('.')]

        # get size of each k set
        self.set_sizes = self.get_set_count()

    def get_set_count(self):

        ksize = round(1 / self.kfold, 2)
        nf = len(self.files)
        size = round(ksize * nf)
        set_sizes = []

        for i in range(4):
            nf = nf - size
            set_sizes.append(size)

        set_sizes.append(nf)
        print(set_sizes)
        return set_sizes

    def distribute_files_randomly(self):

        print()
        print(f'distributing files in {self.kfold} sets')
        print(f'source dir: {os.path.basename(self.source_dir)}')
        print()

        for i in range(self.kfold):
            random.shuffle(self.files)

        start = 0
        end = 0

        for k in range(len(self.set_sizes)):
            end = end + self.set_sizes[k]
            k = k + 1
            outdir = os.path.join(self.target_dir, "k" + str(k))
            create_output_dir(outdir)
            copy_files_to_dir(self.files[start:end], self.source_dir, outdir)
            start = end

        print("---------- DONE ----------")


class FilePicker:
    """
    select n number files randomly from source dir and place in the target dir
    """

    def __init__(self, source_dir, target_dir, nfiles):
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.nfiles = nfiles
        self.files = [f for f in os.listdir(source_dir) if not f.startswith('.')]

    def select_nfiles_randomly(self):
        print()
        print(f'selecting {self.nfiles} files randomly')
        print(f'source dir: {os.path.basename(self.source_dir)}')
        print()

        for i in [1, 2, 3]:
            random.shuffle(self.files)

        cleandir(self.target_dir)
        create_output_dir(self.target_dir)
        copy_files_to_dir(self.files[:self.nfiles], self.source_dir, self.target_dir)

        print("---------- DONE ----------")
