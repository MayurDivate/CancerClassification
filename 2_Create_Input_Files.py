import os

from Data.Sets.utils import create_output_dir
from Data.KfoldInput import KfoldInput
from Data.TrainValTestInput import TrainTestValInput


labels = '/Users/mayurdivate/QUT/Work/Projects/1_CUP/1_DL_data/Labels.txt'
gene_info = 'Data/GeneInfo.tsv'
output_dir = '/Users/mayurdivate/QUT/Work/Projects/1_CUP/2_Data_preprocessing/Input_tsv_files/coding_genes'

data_dir = '/Users/mayurdivate/QUT/Work/Projects/1_CUP/1_DL_data/CNN_CG'
create_output_dir(output_dir)

###########################################################################
# create train test validation data tsv
###########################################################################
def create_test_train_val_tsv():
    train_dir = os.path.join(data_dir, 'Train')
    test_dir = os.path.join(data_dir, 'Test')
    val_dir = os.path.join(data_dir, 'Val')

    train_val_test = TrainTestValInput(train_dir, test_dir, val_dir, labels, gene_info,
                                       genetype='protein_coding',
                                       outdir=output_dir)
    train_val_test.create_train_val_test_tsv()



###########################################################################
# create data for kfold validation tsv
###########################################################################

def create_kfold_tsv():
    k1_dir = os.path.join(data_dir, 'k1')
    k2_dir = os.path.join(data_dir, 'k2')
    k3_dir = os.path.join(data_dir, 'k3')
    k4_dir = os.path.join(data_dir, 'k4')
    k5_dir = os.path.join(data_dir, 'k5')

    Kfold_lincRNA = KfoldInput(inputdirs=[k1_dir, k2_dir, k3_dir, k4_dir, k5_dir],
                               labels_tsv=labels, geneinfo_tsv=gene_info,
                               genetype='protein_coding',
                               outdir=output_dir)

    Kfold_lincRNA.create_k_inputs()

create_test_train_val_tsv()

#create_kfold_tsv()