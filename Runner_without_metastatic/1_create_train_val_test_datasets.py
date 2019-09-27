import os

from Datasets.dl_datasets import Samples


def write_list_file(sample_list, outfile):
    with open(outfile, 'w') as f:
        for sample in sample_list:
            f.write(sample + "\n")

root_dir = '/Users/n10337547/Projects/1_CUP/2_TCGA/0_Data/'
output_dir = '/Users/n10337547/Projects/1_CUP/2_TCGA/2_DL_data/With_metastatic/Sample_list'

sample_dict = {}

for d in os.listdir(root_dir):
    if os.path.isdir(os.path.join(root_dir, d)):
        sample_dict[d] = [f for f in os.listdir(os.path.join(root_dir, d))]


train_list_out = os.path.join(output_dir, 'train_list_wm.txt')
test_list_out = os.path.join(output_dir, 'test_list_wm.txt')
val_list_out = os.path.join(output_dir, 'val_list_wm.txt')

train_list = []
test_list = []
val_list = []

for cancer in sample_dict:
    dl_samples = Samples(sample_dict[cancer])
    train, test, val = dl_samples.get_train_test_val_sets()
    train_list = train_list + train
    test_list = test_list + test
    val_list = val_list + val

write_list_file(train_list, train_list_out)
write_list_file(test_list, test_list_out)
write_list_file(val_list, val_list_out)
