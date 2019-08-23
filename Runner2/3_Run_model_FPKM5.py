import os

from Datasets.utils import create_output_dir
from DeepLearning.train_model import ModelTrainer

data_dir = '/Users/n10337547/Projects/1_CUP/1_Blood_Cancer/2_DL_data/2_BBS_FPKM5/3_Train_Val_Test_mats'

train = os.path.join(data_dir, 'train_mat.tsv')
test = os.path.join(data_dir, 'test_mat.tsv')
val = os.path.join(data_dir, 'val_mat.tsv')


def train_cnn_model():
    resdir = '/Users/n10337547/Projects/1_CUP/1_Blood_Cancer/3_Results/2_FPKM_5_Results'
    mname = 'FPKM_5'
    create_output_dir(resdir)

    cnn_trainer = ModelTrainer(train_tsv=train, test_tsv=test, val_tsv=val, nfeatures=9934, nclasses=5,
                               mname=mname, outdir=resdir)

    model, res = cnn_trainer.run_cnn_model(e=30, test=True)

    model_file = os.path.join(resdir, mname + '_1D_CNN.h5')
    model.save(model_file)
    print('FPKM_5')


train_cnn_model()
