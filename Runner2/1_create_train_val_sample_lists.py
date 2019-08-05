import os

from Datasets.dl_datasets import Samples


def write_list_file(sample_list, outfile):
    with open(outfile, 'w') as f:
        for sample in sample_list:
            f.write(sample + "\n")


# list of samples for each cancer type to divide in train val test sets
#
root_dir = '/Users/n10337547/Projects/1_CUP/1_Blood_Cancer/0_Data/Sample_Lists'
all_samples = os.path.join(root_dir, 'ALL.samples')
aml_samples = os.path.join(root_dir, 'AML.samples')
matureBcell_samples = os.path.join(root_dir, 'Bcell.samples')
breast_samples = os.path.join(root_dir, 'Breast.samples')
skin_samples = os.path.join(root_dir, 'Skin.samples')
metastatic_skin_samples = os.path.join(root_dir, 'Skin_metastatic.samples')
metastatic_breast_samples = os.path.join(root_dir, 'Breast_metastatic.samples')

# output dir for train val test lists
out_dir = '/Users/n10337547/Projects/1_CUP/1_Blood_Cancer/2_DL_data/2_BBS_FPKM5/2_sample_lists/'
train_list_out = os.path.join(out_dir, 'train_list.txt')
test_list_out = os.path.join(out_dir, 'test_list.txt')
val_list_out = os.path.join(out_dir, 'val_list.txt')

train_list = []
test_list = []
val_list = []

samples = [all_samples, aml_samples, matureBcell_samples, breast_samples, skin_samples]

for sample_list_file in samples:
    dl_sample = Samples(sample_list_file)
    train, val, test = dl_sample.get_train_test_val_sets()

    train_list = train_list + train

    test_list = test_list + test
    val_list = val_list + val

meta_skin = [f.rstrip() for f in open(metastatic_skin_samples, 'r')]
meta_breast = [f.rstrip() for f in open(metastatic_breast_samples, 'r')]

print('breast extra : ', len(meta_breast))
print('skin extra: ', len(meta_skin))

meta = meta_breast + meta_skin

test_list = test_list + meta

write_list_file(train_list, train_list_out)
write_list_file(test_list, test_list_out)
write_list_file(val_list, val_list_out)

print("done")
