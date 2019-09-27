from tensorflow.python import keras as ktf

import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




model_file = '/Users/n10337547/Projects/1_CUP/2_TCGA/3_Results/FPKM_1_M/GS_FPKM_1_M_1D_CNN.h5'
test_data = ''
fpkm1_model = ktf.models.load_model(model_file)
g = fpkm1_model.optimizer.get_gradients()
print(g)
exit()

def compute_salient_bases(model, x):
    input_tensors = [model.input]
    gradients = model.optimizer.get_gradients(model.output[0][1], model.input)
    compute_gradients = K.function(inputs=input_tensors, outputs=gradients)

    x_value = np.expand_dims(x, axis=0)
    gradients = compute_gradients([x_value])[0][0]
    sal = np.clip(np.sum(np.multiply(gradients, x), axis=1), a_min=0, a_max=None)
    return sal


sequence_index = 1999  # You can change this to compute the gradient for a different example.
                       # But if so, change the coloring below as well.

sal = compute_salient_bases(model, input_features[sequence_index])

plt.figure(figsize=[16,5])
barlist = plt.bar(np.arange(len(sal)), sal)
[barlist[i].set_color('C1') for i in range(5,17)]  # Change the coloring here if you change the sequence index.
plt.xlabel('Bases')
plt.ylabel('Magnitude of saliency values')
plt.xticks(np.arange(len(sal)), list(sequences[sequence_index]));
plt.title('Saliency map for bases in one of the positive sequences'
          ' (green indicates the actual bases in motif)');

