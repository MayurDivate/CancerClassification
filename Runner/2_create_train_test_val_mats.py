from Datasets.dl_datasets import TrainingTestingData

master_matrix = '/Users/n10337547/Projects/1_CUP/1_Blood_Cancer/2_DL_data/1_BBS_classification/1_Master_Matrices/Blood_breast_skin_master_labeled.tsv'
train_samples= '/Users/n10337547/Projects/1_CUP/1_Blood_Cancer/2_DL_data/1_BBS_classification/2_sample_lists/train_list.txt'
test_samples= '/Users/n10337547/Projects/1_CUP/1_Blood_Cancer/2_DL_data/1_BBS_classification/2_sample_lists/test_list.txt'
val_samples= '/Users/n10337547/Projects/1_CUP/1_Blood_Cancer/2_DL_data/1_BBS_classification/2_sample_lists/val_list.txt'
outdir = '/Users/n10337547/Projects/1_CUP/1_Blood_Cancer/2_DL_data/1_BBS_classification/3_Train_Val_Test_mats'



train_val_test_data = TrainingTestingData(master_matrix, train_samples, val_samples,test_samples, outdir)
train_val_test_data.create_dl_datasets()


