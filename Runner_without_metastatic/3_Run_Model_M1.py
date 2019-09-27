import os

from Datasets.utils import create_output_dir
from DeepLearning.train_model import ModelTrainer


def train_model():
    print(train)
    print('Model ')
    create_output_dir(resdir)

    model_trainer = ModelTrainer(train_tsv=train, test_tsv=test, val_tsv=val, nfeatures=ngenes,
                                 labels_file=labels, mname=mname, outdir=resdir)

    print('training started')
    model, res = model_trainer.run_cnn_model(e=25, test=True)
    model_file = os.path.join(resdir, mname + '_1D_CNN.h5')

    # model, res = model_trainer.run_mlp_model(e=30, test=True)
    # model_file = os.path.join(resdir, mname + '_MLP.h5')

    model.save(model_file)
    print('done')


# ## define other variables

data_dir = '/Users/n10337547/Projects/1_CUP/2_TCGA/2_DL_data/3_Train_val_test_mats'
labels = '/Users/n10337547/Projects/1_CUP/2_TCGA/0_Data/TCGA_classes.txt'

resdir = '/Users/n10337547/Projects/1_CUP/2_TCGA/3_Results/FPKM_1_M'
mname = 'GS_FPKM_1_M'
ngenes = 893    

# F5 1515, F1 893

train = os.path.join(data_dir, 'train_FPKM_1_M.tsv')
test = os.path.join(data_dir, 'test_FPKM_1_M.tsv')
val = os.path.join(data_dir, 'val_FPKM_1_M.tsv')

train_model()
