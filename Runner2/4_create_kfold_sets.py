import os

from Datasets.dl_datasets import Kfold_sets

root_dir = "/Users/n10337547/Projects/1_CUP/1_Blood_Cancer/0_Data/Sample_Lists"
cancer_files = ['ALL.samples', 'Bcell.samples', 'AML.samples', 'Breast.samples', 'Skin.samples']
outdir = '/Users/n10337547/Projects/1_CUP/1_Blood_Cancer/0_Data/k_fold_lists'

for cancer in cancer_files:
    print(cancer_files)
    K5_sets = Kfold_sets(5, os.path.join(root_dir, cancer), outdir, suffix=cancer)
    K5_sets.distribute_files_randomly()

