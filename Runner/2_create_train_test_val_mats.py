from Datasets import train_test_val_matrices as tvt

master_matrix = '/Users/n10337547/Projects/1_CUP/1_Blood_Cancer/2_DL_data/1_BBS_classification/1_Master_Matrices/Blood_breast_skin_master_labeled.tsv'
train_list_out = '/Users/n10337547/Projects/1_CUP/1_Blood_Cancer/2_DL_data/1_BBS_classification/2_sample_lists/train_list.txt'
test_list_out = '/Users/n10337547/Projects/1_CUP/1_Blood_Cancer/2_DL_data/1_BBS_classification/2_sample_lists/test_list.txt'
val_list_out = '/Users/n10337547/Projects/1_CUP/1_Blood_Cancer/2_DL_data/1_BBS_classification/2_sample_lists/val_list.txt'
outdir = '/Users/n10337547/Projects/1_CUP/1_Blood_Cancer/2_DL_data/1_BBS_classification/3_Train_Val_Test_mats'

tvt.get_train_test_val_matrices(master_matrix, train_list_out, val_list_out, test_list_out, outdir)



