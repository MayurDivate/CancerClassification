import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder


label_file = '/Users/n10337547/Projects/1_CUP/3_Primary_tumor_classification/0_Data/TCGA_classes.txt'

labels = np.array([label.strip() for label in open(label_file)]).reshape(-1,1)
ohe = OneHotEncoder()
ohe.fit(labels)
encoded = ohe.transform(labels).toarray()


df = pd.DataFrame([labels, encoded])
print(df.T.to_csv('test.csv', sep='\t'))

