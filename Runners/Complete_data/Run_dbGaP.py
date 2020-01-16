from DeepLearning.load_and_run_model import PretrainedModel

model_file = '/Users/n10337547/Projects/1_CUP/3_Primary_tumor_classification/2_results/1_log_transformed_models/FPKM_5_1D_CNN.h5'
label_file = '/Users/n10337547/Projects/1_CUP/3_Primary_tumor_classification/0_Data/TCGA_classes.txt'

data_file = '/Users/n10337547/Projects/1_CUP/4_dbGaP_MET500/3_Data_preprocessing/1_without_mt_genes/1_mats/dbGaP_FPKM_5.tsv'
outfile = '/Users/n10337547/Projects/1_CUP/4_dbGaP_MET500/4_Results/1_LogTransformed_models/dbGaP_FPKM_5_pred.tsv'

dbgap = PretrainedModel(model_file, data_file, outfile, label_file)

dbgap.load_and_run_model()
