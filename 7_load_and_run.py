import os

from Data.InputData import InputData
from Evaluation.evaluater import ModelEvaluater

labels = '/Users/mayurdivate/QUT/Work/Projects/1_CUP/1_DL_data/Labels.txt'

test_dir = "/Users/mayurdivate/QUT/Work/Projects/1_CUP/1_DL_data/CNN_CG/SKCM_metastatic_testing"

gene_info = 'Data/GeneInfo.tsv'

outdir = '/Users/mayurdivate/QUT/Work/Projects/1_CUP/2_Data_preprocessing/Input_tsv_files/coding_genes'
outfile = 'Skin_Metastatic.tsv'
model_file = '/Users/mayurdivate/QUT/Work/Projects/1_CUP/3_Results/Cg_Skin_Cnn/1_Cg_skin_1D_CNN_model.h5'

TestData = InputData(test_dir, labels, gene_info, 'protein_coding', outdir)
TestData.create_input_tsv(TestData.inputdir, outfile)

input_tsv = os.path.join(outdir, outfile)
results = '/Users/mayurdivate/QUT/Work/Projects/1_CUP/3_Results/Skin_Metastatic_testing/test.txt'
model_eval = ModelEvaluater(model_file, input_tsv, nfeatures=19814, outfile=results)

model_eval.run_evaluation()

print('- - - - - - Finished - - - - - -')