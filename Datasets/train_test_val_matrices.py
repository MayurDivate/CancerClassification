from Matrix.filters import SampleFilter
import os

def get_train_test_val_matrices(master_mat, train_samples, val_samples, test_samples, outdir):

    train = SampleFilter(master_mat, train_samples, os.path.join(outdir, 'train_mat.tsv'))
    test = SampleFilter(master_mat, test_samples, os.path.join(outdir, 'test_mat.tsv'))
    val = SampleFilter(master_mat, val_samples, os.path.join(outdir,'val_mat.tsv'))


    train.filter()
    print(train_samples)
    val.filter()
    print(val_samples)
    test.filter()
    print(test_samples)


