import os

from Data.Sets.utils import create_output_dir
from DeepLearning.kfold_validation import K5FoldCrossValidation
from DeepLearning.train_model import TrainTestValidate

### train and save model

# data_dir = '/Users/n10337547/Projects/1_CUP/1_Blood_Cancer/3_Data_preprocessing/Input_tsv_files/lincRNA_1226'
# data_dir = '/Users/n10337547/Projects/1_CUP/1_Blood_Cancer/3_Data_preprocessing/Input_tsv_files/coding_genes'
data_dir = '/Users/mayurdivate/QUT/Work/Projects/1_CUP/2_Data_preprocessing/Input_tsv_files/coding_genes'


def mlp_model():
    resdir = '/Users/mayurdivate/QUT/Work/Projects/1_CUP/3_Results/Cg_Skin_MLP'
    create_output_dir(resdir)

    train = os.path.join(data_dir, 'Train.tsv')
    test = os.path.join(data_dir, 'Test.tsv')
    val = os.path.join(data_dir, 'Val.tsv')

    ModelTrainer = TrainTestValidate(train_tsv=train, test_tsv=test, val_tsv=val,
                                     nfeatures=19814, nclasses=5, mname="Cg_Skin", outdir=resdir)

    model, res = ModelTrainer.run_mlp_model(e=20)

    model_file = ModelTrainer.outdir + "/Cg_skin_mlp.h5"
    model.save(model_file)


def cnn_model():
    resdir = '/Users/mayurdivate/QUT/Work/Projects/1_CUP/3_Results/Cg_Skin_Cnn'
    create_output_dir(resdir)

    train = os.path.join(data_dir, 'Train.tsv')
    test = os.path.join(data_dir, 'Test.tsv')
    val = os.path.join(data_dir, 'Val.tsv')

    ModelTrainer = TrainTestValidate(train_tsv=train, test_tsv=test, val_tsv=val,
                                     nfeatures=19814, nclasses=5,
                                     mname="Cg_Skin", outdir=resdir)

    model, res = ModelTrainer.run_cnn_model(e=25)

    model_file = ModelTrainer.outdir + "/Cg_skin_1D_CNN_model.h5"
    model.save(model_file)


# ------------------------------
# kfold validaion

def kfold_validation():
    k1 = os.path.join(data_dir, 'K1.tsv')
    k2 = os.path.join(data_dir, 'K2.tsv')
    k3 = os.path.join(data_dir, 'K3.tsv')
    k4 = os.path.join(data_dir, 'K4.tsv')
    k5 = os.path.join(data_dir, 'K5.tsv')

    kfold_outdir = '/Users/mayurdivate/QUT/Work/Projects/1_CUP/3_Results/Cg_Skin_kfold'

    lincRNA_k5 = K5FoldCrossValidation(k1, k2, k3, k4, k5, mname='CNN',
                                       nfeatures=19814, nclasses=5, outdir=kfold_outdir)
    lincRNA_k5.run_kfold_cross_validation(25)


# mlp_model()

kfold_validation()

# cnn_model()
