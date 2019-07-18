import os

from Data.Sets.DLsets import TrainValTestSets, Kfold_sets, FilePicker
from Data.Sets.utils import cleandir

# #############
# input dirs
# ############

data_dir = '/Users/mayurdivate/QUT/Work/Projects/1_CUP/0_Data'

breast_complete_dir = os.path.join(data_dir, 'Complete_TCGA_Breast')
breast_dir = os.path.join(data_dir, 'Breast')
all_dir = os.path.join(data_dir, 'ALL')
aml_dir = os.path.join(data_dir, 'AML')
bcell_dir = os.path.join(data_dir, 'Bcell')

output_dir = '/Users/mayurdivate/QUT/Work/Projects/1_CUP/1_DL_data/lincRNA'


def create_train_val_test_objects(outdir):
    """
    # specific for blood cancer data
    :param outdir:
    :return:
    """
    All = TrainValTestSets(source_dir=all_dir, target_dir=outdir)
    Aml = TrainValTestSets(source_dir=aml_dir, target_dir=outdir)
    Bcell = TrainValTestSets(source_dir=bcell_dir, target_dir=outdir)
    Breast = TrainValTestSets(source_dir=breast_dir, target_dir=outdir)
    return All, Aml, Bcell, Breast


def create_kfold_objects(outdir):
    """
    # objects for kfold validation
    :param outdir:
    :return:
    """
    All_k5 = Kfold_sets(source_dir=all_dir, target_dir=output_dir, kfold=5)
    Aml_k5 = Kfold_sets(source_dir=aml_dir, target_dir=output_dir, kfold=5)
    Bcell_k5 = Kfold_sets(source_dir=bcell_dir, target_dir=output_dir, kfold=5)
    Breast_K5 = Kfold_sets(source_dir=breast_dir, target_dir=output_dir, kfold=5)
    return All_k5, Aml_k5, Bcell_k5, Breast_K5


def select_500_breast_files():
    """
    Randomly select 500 breast cancer samples
    """
    BreastCancerSamplesPicker = FilePicker(breast_complete_dir, breast_dir, 500)
    BreastCancerSamplesPicker.select_nfiles_randomly()


def create_train_test_val_dataset(out):
    """
    create Train val And Test data sets
    """
    cleandir(out)

    All, Aml, Bcell, Breast = create_train_val_test_objects(out)

    All.distribute_files_randomly()
    Aml.distribute_files_randomly()
    Bcell.distribute_files_randomly()
    Breast.distribute_files_randomly()


def kfold_datasets(out):
    """
    create Kfold sets
    """
    All_k5, Aml_k5, Bcell_k5, Breast_K5 = create_kfold_objects(out)

    All_k5.distribute_files_randomly()
    Aml_k5.distribute_files_randomly()
    Bcell_k5.distribute_files_randomly()
    Breast_K5.distribute_files_randomly()


output_dir = '/Users/mayurdivate/QUT/Work/Projects/1_CUP/1_DL_data/CNN_CG'

select_500_breast_files()

create_train_test_val_dataset(output_dir)

#kfold_datasets(output_dir)
