import shap
import numpy as np
import pandas as pd
from tensorflow.python.keras import models

model_file = '/Users/n10337547/Projects/1_CUP/3_Primary_tumor_classification/2_results/1_log_transformed_models/FPKM_5_1D_CNN.h5'
data_file = '/Users/n10337547/Projects/1_CUP/3_Primary_tumor_classification/1_DL_Data/TCGA_primary_cancers_FPKM_5.csv'

model = models.load_model(model_file)

data = pd.read_csv(data_file, sep='\t', nrows=10, index_col=0)
dataX = np.log10(data.iloc[:, :-1]+1)
dataY = data.iloc[:, -1]

print(dataY.head())
print(dataX.head())



explainer = shap.TreeExplainer()
