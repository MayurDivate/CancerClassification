from Datasets.dl_datasets import TrainingTestingData
import os

root_dir = '/Users/n10337547/Projects/1_CUP/2_TCGA/2_DL_data/2_samples/'
master_matrix = '/Users/n10337547/Projects/1_CUP/2_TCGA/2_DL_data/1_master_matrices/FPKM_1_GS_master.tsv'

train_samples = os.path.join(root_dir, 'train_list.txt')
test_samples = os.path.join(root_dir,'test_list.txt')
val_samples = os.path.join(root_dir,'val_list.txt')

outdir = '/Users/n10337547/Projects/1_CUP/2_TCGA/2_DL_data/3_Train_val_test_mats'


train_val_test_data = TrainingTestingData(master_matrix, train_samples, val_samples, test_samples, outdir)
train_val_test_data.create_dl_datasets()


