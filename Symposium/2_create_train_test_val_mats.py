from Datasets.dl_datasets import TrainingTestingData
import os


def go(root_dir, master_matrix, outdir):
    train_samples = os.path.join(root_dir, 'train_list.txt')
    test_samples = os.path.join(root_dir,'test_list.txt')
    val_samples = os.path.join(root_dir,'val_list.txt')

    train_val_test_data = TrainingTestingData(master_matrix, train_samples, val_samples, test_samples, outdir)
    train_val_test_data.create_dl_datasets()
    print("- -- --- DONE --- -- -")

## BBS 1

rdir= '/Users/n10337547/Projects/3_Symposium/2_DL_Data/BBS/2_samples'
mat = '/Users/n10337547/Projects/3_Symposium/2_DL_Data/BBS/1_labeled_mat/BBS_FPKM_1_CG_labeled_master.tsv'
out = '/Users/n10337547/Projects/3_Symposium/2_DL_Data/BBS/3_Train_Val_Test_mats/1'

go(rdir, mat, out)

## BBS 5

rdir= '/Users/n10337547/Projects/3_Symposium/2_DL_Data/BBS/2_samples'
mat = '/Users/n10337547/Projects/3_Symposium/2_DL_Data/BBS/1_labeled_mat/BBS_FPKM_5_CG_labeled_master.tsv'
out = '/Users/n10337547/Projects/3_Symposium/2_DL_Data/BBS/3_Train_Val_Test_mats/5'

go(rdir, mat, out)

## without meta 1

rdir= '/Users/n10337547/Projects/3_Symposium/2_DL_Data/without_metastatic/2_samples'
mat = '/Users/n10337547/Projects/3_Symposium/2_DL_Data/without_metastatic/1_labeled_mat/FPKM_1_CG_master_labeled.tsv'
out = '/Users/n10337547/Projects/3_Symposium/2_DL_Data/without_metastatic/3_Train_Val_Test_mats/1'

go(rdir, mat, out)

## without meta 5

rdir= '/Users/n10337547/Projects/3_Symposium/2_DL_Data/without_metastatic/2_samples'
mat = '/Users/n10337547/Projects/3_Symposium/2_DL_Data/without_metastatic/1_labeled_mat/FPKM_5_CG_master_labeled.tsv'
out = '/Users/n10337547/Projects/3_Symposium/2_DL_Data/without_metastatic/3_Train_Val_Test_mats/5'

go(rdir, mat, out)

## with meta 1
rdir= '/Users/n10337547/Projects/3_Symposium/2_DL_Data/with_metastatic/2_samples'
mat = '/Users/n10337547/Projects/3_Symposium/2_DL_Data/with_metastatic/1_labeled_mat/FPKM_1_CG_master_labeled.tsv'
out = '/Users/n10337547/Projects/3_Symposium/2_DL_Data/with_metastatic/3_Train_Val_Test_mats/1'

go(rdir, mat, out)

## with meta 5
rdir= '/Users/n10337547/Projects/3_Symposium/2_DL_Data/with_metastatic/2_samples'
mat = '/Users/n10337547/Projects/3_Symposium/2_DL_Data/with_metastatic/1_labeled_mat/FPKM_5_CG_master_labeled.tsv'
out = '/Users/n10337547/Projects/3_Symposium/2_DL_Data/with_metastatic/3_Train_Val_Test_mats/5'

go(rdir, mat, out)