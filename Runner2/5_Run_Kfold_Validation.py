from DeepLearning.kfold import KFoldCrossValidation
import os

outdir = '/Users/n10337547/Projects/1_CUP/1_Blood_Cancer/0_Data/k_fold_lists'
k_list = [os.path.join(outdir,'K1.txt'),
          os.path.join(outdir,'K2.txt'),
          os.path.join(outdir,'K3.txt'),
          os.path.join(outdir,'K4.txt'),
          os.path.join(outdir,'K5.txt')]

master_matrix = '/Users/n10337547/Projects/1_CUP/1_Blood_Cancer/2_DL_data/1_BBS_classification/1_Master_Matrices/Blood_breast_skin_master_labeled.tsv'
#nfeatures = 9934
nfeatures = 13565
mname = 'Blood Breast Skin Model 6'
outdir = '/Users/n10337547/Projects/1_CUP/1_Blood_Cancer/3_Results/1_BBS_classification/kfold_cross_validation'

k5_validation = KFoldCrossValidation(k_list, master_matrix, mname, nfeatures, 5, outdir)
k5_validation.run_kfold_cross_validation()