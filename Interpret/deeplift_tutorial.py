import deeplift
from deeplift.conversion import kerasapi_conversion as kc
import keras as ks


model_file = '/Users/n10337547/Projects/3_Symposium/3_Results/keras_without_metastatic/WM_FPKM_5_1D_CNN.h5'

kmodel = ks.models.load_model(model_file)

#deeplift_model = kc.convert_model_from_saved_files(model_file)

