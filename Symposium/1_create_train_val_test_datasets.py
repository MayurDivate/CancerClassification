import os
from Datasets.dl_datasets import Samples


def write_list_file(sample_list, outfile):
    with open(outfile, 'w') as f:
        for sample in sample_list:
            f.write(sample + "\n")


root_dir = '/Users/n10337547/Projects/2_Cancer_Genes/0_Data/'

sample_dict = {}

for d in os.listdir(root_dir):

     if os.path.isdir(os.path.join(root_dir,d)):
          sample_dict[d] = [f for f in os.listdir(os.path.join(root_dir,d))]


output_dir = '/Users/n10337547/Projects/3_Symposium/1_Data_preprocessing/without_metastatic/2_samples'
train_list_out = os.path.join(output_dir, 'train_list.txt')
test_list_out = os.path.join(output_dir, 'test_list.txt' )
val_list_out = os.path.join(output_dir, 'val_list.txt' )

train_list = []
test_list = []
val_list = []

for cancer in sample_dict:
     dl_samples = Samples(sample_dict[cancer])
     train, test, val = dl_samples.get_train_test_val_sets()
     train_list = train_list + train
     test_list = test_list + test
     val_list = val_list + val

exit()

write_list_file(train_list, train_list_out)
write_list_file(test_list, test_list_out)
write_list_file(val_list, val_list_out)

