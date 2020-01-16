import os
from DeepLearning.train_model import ModelTrainer
from Results.classification_report import ClassificationReport

def trainCNN():
    inputfile = '/Users/n10337547/Projects/1_CUP/3_Primary_tumor_classification/0_Data/FPKM_5_CG_labeled_master.tsv'
    label_file = '/Users/n10337547/Projects/1_CUP/3_Primary_tumor_classification/0_Data/TCGA_classes.txt'
    resdir = '/Users/n10337547/Projects/1_CUP/3_Primary_tumor_classification/2_results'
    mname = 'FPKM_5'

    #model_trainer = ModelTrainer(inputfile, label_file,mname, resdir)
    #model, res = model_trainer.run_cnn_model(e=25, test=True)
    #model_file = os.path.join(resdir, mname + '_1D_CNN.h5')
    #model.save(model_file)

    model_res_file = os.path.join(resdir, mname + "_1D_CNN_predict.txt")
    #model_res_file = os.path.join(resdir, mname + "_confusion_matrix.txt")
    ClassificationReport(model_res_file, 78, label_file, 79, resdir).parse_results()

    print(res)

trainCNN()