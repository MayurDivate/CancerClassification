import os

from Datasets.utils import create_output_dir
from DeepLearning.train_model import ModelTrainer

data_dir = '/Users/n10337547/Projects/1_CUP/2_TCGA/2_DL_data/3_Train_val_test_mats'
labels = '/Users/n10337547/Projects/1_CUP/2_TCGA/0_Data/TCGA_classes.txt'

train = os.path.join(data_dir, 'train_FPKM_1_mat.tsv')
test = os.path.join(data_dir, 'test_FPKM_1_mat.tsv')
val = os.path.join(data_dir, 'val_FPKM_1_mat.tsv')

resdir = '/Users/n10337547/Projects/1_CUP/2_TCGA/3_Results/FPKM_1_models'

def train_model():

    mname = 'FPKM_1'
    create_output_dir(resdir)
    model_trainer = ModelTrainer(train_tsv=train, test_tsv=test, val_tsv=val, nfeatures=893,
                               labels_file=labels, mname=mname, outdir=resdir)

    model, res = model_trainer.run_mlp_model(e=30, test=True)
    model_file = os.path.join(resdir, mname + '_MLP.h5')
    model.save(model_file)


train_model()
