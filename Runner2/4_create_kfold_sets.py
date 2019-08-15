from Datasets.dl_datasets import Kfold_sets

file = '/Users/n10337547/Projects/1_CUP/1_Blood_Cancer/2_DL_data/2_BBS_FPKM5/2_sample_lists/m.txt'
outdir = '.'
K5_sets = Kfold_sets(5, file, outdir)

K5_sets.distribute_files_randomly()