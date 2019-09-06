import os

from Datasets.dl_datasets import TrainingTestingData

root_dir = '/Users/n10337547/Projects/3_Symposium/2_DL_Data/without_metastatic/2_samples/'
master_matrix = '/Users/n10337547/Projects/3_Symposium/2_DL_Data/without_metastatic/1_labeled_mat/FPKM_5_CG_master_labeled.tsv'

train_samples = os.path.join(root_dir, 'train_list.txt')
test_samples = os.path.join(root_dir, 'test_list.txt')
val_samples = os.path.join(root_dir, 'val_list.txt')

outdir = '/Users/n10337547/Projects/3_Symposium/2_DL_Data/without_metastatic/3_Train_Val_Test_mats'

train_val_test_data = TrainingTestingData(master_matrix, train_samples, val_samples, test_samples, outdir)
train_val_test_data.create_dl_datasets()

