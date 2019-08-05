from Datasets.dl_datasets import TrainingTestingData
import os

root_dir = '/Users/n10337547/Projects/1_CUP/1_Blood_Cancer/2_DL_data/2_BBS_FPKM5/2_sample_lists/'
master_matrix = '/Users/n10337547/Projects/1_CUP/1_Blood_Cancer/2_DL_data/2_BBS_FPKM5/1_Master_Matrices/Blood_breast_skin_master_labeled_FPKM_5.tsv'
train_samples = os.path.join(root_dir, 'train_list.txt')
test_samples = os.path.join(root_dir,'test_list.txt')
val_samples = os.path.join(root_dir,'val_list.txt')
outdir = '/Users/n10337547/Projects/1_CUP/1_Blood_Cancer/2_DL_data/2_BBS_FPKM5/3_Train_Val_Test_mats'



train_val_test_data = TrainingTestingData(master_matrix, train_samples, val_samples,test_samples, outdir)
train_val_test_data.create_dl_datasets()


