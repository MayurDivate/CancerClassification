from DeepLearning.kfold import KFoldCrossValidation
import os

outdir = '/Users/n10337547/Projects/1_CUP/1_Blood_Cancer/0_Data/k_fold_lists'
k_list = [os.path.join(outdir,'K1.txt'),
          os.path.join(outdir,'K2.txt'),
          os.path.join(outdir,'K3.txt'),
          os.path.join(outdir,'K4.txt'),
          os.path.join(outdir,'K5.txt')]

nfeatures = 9934
mname = 'K5 Blood Breast Skin'
outdir = '/Users/n10337547/Projects/1_CUP/1_Blood_Cancer/3_Results/2_FPKM_5_Results/kfold_cross_validation'

k5_validation = KFoldCrossValidation(k_list,mname, nfeatures, 5, outdir)
k5_validation.run_kfold_cross_validation()