from DeepLearning.load_and_run_model import PretrainedModel

dbgap = PretrainedModel(
        model_file = '/Users/n10337547/Projects/3_Symposium/3_Results/without_metastatic/WM_FPKM_5_1D_CNN.h5',
        data_file = '/Users/n10337547/Projects/1_CUP/3_dbGaP/3_FPKM_mats/dbGaP_FPKM_5_master.tsv',
        nfeatures = 13488,
        outfile = '/Users/n10337547/Projects/1_CUP/3_dbGaP/3_FPKM_mats/dbGaP_FPKM_5_pred.tsv',)


dbgap.load_and_run_model()

