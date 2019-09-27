import os

from Datasets.utils import create_output_dir
from DeepLearning.train_model import ModelTrainer


class ReducedModel:

    def __init__(self, data_dir, resdir, mname, nfeatures):
        self.data_dir = data_dir
        self.resdir = resdir
        self.mname = mname
        self.nfeatures = nfeatures

    def run_reduced_mlp_model(self):
        create_output_dir(self.resdir)

        train = os.path.join(self.data_dir, 'Train.tsv')
        test = os.path.join(self.data_dir, 'Test.tsv')
        val = os.path.join(self.data_dir, 'Val.tsv')

        model_trainer = ModelTrainer(train_tsv=train, test_tsv=test, val_tsv=val,
                                    nfeatures=self.nfeatures, mname=self.mname, outdir=self.resdir)

        model, res = model_trainer.run_mlp_model()

        model_file = os.path.join(ModelTrainer.outdir, self.mname + ".h5")

        model.save(model_file)
        return res

    def run_reduced_cnn_model(self):
        create_output_dir(self.resdir)

        train = os.path.join(self.data_dir, 'Train.tsv')
        test = os.path.join(self.data_dir, 'Test.tsv')
        val = os.path.join(self.data_dir, 'Val.tsv')

        model_trainer = ModelTrainer(train_tsv=train, test_tsv=test, val_tsv=val,
                                    nfeatures=self.nfeatures, mname=self.mname,
                                    outdir=self.resdir)

        model, res = model_trainer.run_cnn_model()

        model_file = os.path.join(ModelTrainer.outdir, self.mname + ".h5")

        model.save(model_file)
        return res