from tensorflow.python import keras as ktf

import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DeepLearning.data_preprocess import Preprocessor


model_file = '/Users/n10337547/Projects/1_CUP/2_TCGA/3_Results/FPKM_1_M/GS_FPKM_1_M_1D_CNN.h5'
labels = '/Users/n10337547/Projects/1_CUP/2_TCGA/0_Data/TCGA_classes.txt'
data_file = '/Users/n10337547/Projects/1_CUP/2_TCGA/2_DL_data/3_Train_val_test_mats/val_FPKM_1_M.tsv'

fpkm1_model = ktf.models.load_model(model_file)

def get_model_input(data_file, nfeatures, data_labels):
    data_preprocessor = Preprocessor([data_file],nfeatures, data_labels)
    return data_preprocessor.get_cnn_data()

def compute_salient_bases(model, x):

    input_tensors = [model.input]

    with tf.GradientTape as tape:
        loss = tf.losses.categorical_crossentropy
        gradients = model.optimizer.get_gradients(model.output[0][1], model.input)
        print(gradients)

    exit()
    compute_gradients = K.function(inputs=input_tensors, outputs=gradients)

    x_value = np.expand_dims(x, axis=0)
    gradients = compute_gradients([x_value])[0][0]
    sal = np.clip(np.sum(np.multiply(gradients, x), axis=1), a_min=0, a_max=None)
    return sal


print(" Calculate silency mapping")
sequence_index = 1000
input_data, input_labels = get_model_input(data_file, 893, labels)
sal = compute_salient_bases(fpkm1_model, input_data[sequence_index])

exit()

"""
plt.figure(figsize=[16,5])
barlist = plt.bar(np.arange(len(sal)), sal)
[barlist[i].set_color('C1') for i in range(5,17)]  # Change the coloring here if you change the sequence index.
plt.xlabel('Bases')
plt.ylabel('Magnitude of saliency values')
plt.xticks(np.arange(len(sal)), list(sequences[sequence_index]));
plt.title('Saliency map for bases in one of the positive sequences'
          ' (green indicates the actual bases in motif)');

"""
