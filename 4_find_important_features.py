import os

from Interpret.model_reader import MLPmodelReader, CNNmodleReader


def run_feature_scoring(model_name):
    print('MODEL:', model_name)
    basedir = '/Users/n10337547/Projects/1_CUP/1_Blood_Cancer/4_Results/lincRNA_MLP'

    model_file = os.path.join(basedir, model_name)

    mlp_model_reader = MLPmodelReader(model_file)

    feature_scores = mlp_model_reader.get_feature_importance_score()

    output_file = model_file.replace('.h5', '_feature_scores.tsv')

    mlp_model_reader.write_scores_csv(feature_scores, output_file)


models = ['1_lincRNA_mlp.h5', '2_lincRNA_mlp.h5', '3_lincRNA_mlp.h5', '4_lincRNA_mlp.h5']


def run_cnn_feature_scoring(model_name):
    print('MODEL:', model_name)
    basedir = '/Users/mayurdivate/QUT/Work/Projects/1_CUP/3_Results/CNN_CG'

    model_file = os.path.join(basedir, model_name)

    CgCnnReader = CNNmodleReader(model_file)

    feature_scores = CgCnnReader.get_feature_importance_score()

    output_file = model_file.replace('.h5', '_feature_scores.tsv')

    CgCnnReader.write_scores_csv(feature_scores, output_file)


run_cnn_feature_scoring('3_coding_genes_1D_CNN_model.h5')

cnn_models = ['3_coding_genes_1D_CNN_model.h5',
              '4_coding_genes_1D_CNN_model.h5',
              '5_coding_genes_1D_CNN_model.h5']

for model in cnn_models:
    run_cnn_feature_scoring(model)

