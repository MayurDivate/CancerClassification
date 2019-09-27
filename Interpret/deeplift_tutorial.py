import deeplift
import tensorflow as tf
from deeplift.conversion import kerasapi_conversion as kc
import keras as k


model_file = '/Users/n10337547/Projects/1_CUP/2_TCGA/3_Results/FPKM_1_M/GS_FPKM_1_M_1D_CNN.h5'
kmodel_file = 'kmodel.h5'

kmodel = tf.keras.models.load_model(model_file)

#tf.keras.models.save_model(kmodel, kmodel_file)
#k.models.save_model(kmodel, kmodel_file)

#deeplift_model = kc.convert_model_from_saved_files(model_file)

