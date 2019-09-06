import os

from Datasets.utils import create_output_dir
from DeepLearning.train_model import ModelTrainer

data_dir = '/Users/n10337547/Projects/3_Symposium/2_DL_Data/BBS/3_Train_Val_Test_mats'
labels = '/Users/n10337547/Projects/1_CUP/2_TCGA/0_Data/TCGA_classes.txt'

resdir = '/Users/n10337547/Projects/3_Symposium/3_Results/1_BBS/'

mname = 'BBS_FPKM_5'
ngenes = 13488

# meta 16406, 13488
# without meta

train = os.path.join(data_dir, 'train_BBS_5_mat.tsv')
test = os.path.join(data_dir, 'test_BBS_5_mat.tsv')
val = os.path.join(data_dir, 'val_BBS_5_mat.tsv')


def train_model():

    print('Model ')
    create_output_dir(resdir)

    model_trainer = ModelTrainer(train_tsv=train, test_tsv=test, val_tsv=val, nfeatures=ngenes,
                               labels_file=labels, mname=mname, outdir=resdir)

    print('training started')
    model, res = model_trainer.run_cnn_model(e=25, test=True)
    model_file = os.path.join(resdir, mname + '_1D_CNN.h5')

    #model, res = model_trainer.run_mlp_model(e=30, test=True)
    #model_file = os.path.join(resdir, mname + '_MLP.h5')

    model.save(model_file)
    print('done')

#train_model()

mname = 'BBS_FPKM_1'
ngenes = 16406

train = os.path.join(data_dir, 'train_BBS_1_mat.tsv')
test = os.path.join(data_dir, 'test_BBS_1_mat.tsv')
val = os.path.join(data_dir, 'val_BBS_1_mat.tsv')

train_model()
