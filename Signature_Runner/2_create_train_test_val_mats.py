from Datasets.dl_datasets import TrainingTestingData
import os

root_dir = '/Users/n10337547/Projects/1_CUP/1_Blood_Cancer/6_Based_Gene_Signatures/1_DL_Data/FPKM_5/'
master_matrix = os.path.join(root_dir, '1_Master_Matrix/FPKM_5_Master_gene_signature_matrix.tsv')

train_samples = os.path.join(root_dir, '2_sample_lists/train_list.txt')
test_samples = os.path.join(root_dir,'2_sample_lists/test_list.txt')
val_samples = os.path.join(root_dir,'2_sample_lists/val_list.txt')

outdir = os.path.join(root_dir, '3_Train_Val_Test_mats')

train_val_test_data = TrainingTestingData(master_matrix, train_samples, val_samples,test_samples, outdir)
train_val_test_data.create_dl_datasets()

# ==== ==== === == == == = = === == === ===
print("FPKM 1")

root_dir = '/Users/n10337547/Projects/1_CUP/1_Blood_Cancer/6_Based_Gene_Signatures/1_DL_Data/FPKM_1'
master_matrix = os.path.join(root_dir, '1_Master_Matrix/FPKM_1_Master_gene_signature_matrix.tsv')

train_samples = os.path.join(root_dir, '2_sample_lists/train_list.txt')
test_samples = os.path.join(root_dir,'2_sample_lists/test_list.txt')
val_samples = os.path.join(root_dir,'2_sample_lists/val_list.txt')
outdir = os.path.join(root_dir, '3_Train_Val_Test_mats')


train_val_test_data = TrainingTestingData(master_matrix, train_samples, val_samples,test_samples, outdir)
train_val_test_data.create_dl_datasets()

