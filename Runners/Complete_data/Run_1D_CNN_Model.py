import os
from DeepLearning.train_model import ModelTrainer
from Results.classification_report import ClassificationReport

def trainCNN():

    root_dir =  '/Users/n10337547/Projects/1_CUP/3_Primary_tumor_classification/'
    data_dir = os.path.join(root_dir, '1_DL_Data')
    inputfile = os.path.join(data_dir, 'TCGA_primary_cancers_FPKM_5.csv')
    label_file = os.path.join(data_dir, 'TCGA_classes.txt')

    resdir = os.path.join(root_dir,'3_CNN_results/CNN_tf')
    mname = 'FPKM_5'
    nfeatures = 13471

    model_trainer = ModelTrainer(inputfile, label_file, mname, nfeatures, resdir)
    model, res = model_trainer.run_cnn_model(e=25, test=True)
    model_file = os.path.join(resdir, mname + '_1D_CNN.h5')
    model.save(model_file)

    model_res_file = os.path.join(resdir, mname + "_1D_CNN_predict.txt")
    report_file = os.path.join(resdir, mname + "_confusion_matrix.txt")
    ClassificationReport(model_res_file, 78, label_file, 79, report_file).parse_results()

    print(res)

trainCNN()