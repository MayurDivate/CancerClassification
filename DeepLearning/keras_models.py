import keras as ks
import numpy as np

np.random.seed(3)

class kmodel:

    def __init__(self, nfeatures, nclasses):
        self.nfeatures = nfeatures
        self.nclasses = nclasses

    def get_1D_cnn_model(self):
        np.random.seed(7)

        model = ks.Sequential()

        model.add(ks.layers.Conv1D(10, kernel_size=1, activation=ks.activations.relu(), input_shape=(1, self.nfeatures)))
        model.add(ks.layers.Flatten())
        model.add(ks.layers.Dense(100, activation=ks.activations.relu()))
        model.add(ks.layers.Dense(100, activation=ks.activations.relu()))
        model.add(ks.layers.Dense(100, activation=ks.activations.relu()))
        model.add(ks.layers.Dense(100, activation=ks.activations.relu()))
        model.add(ks.layers.Dense(100, activation=ks.activations.relu()))

        model.add(ks.layers.Dense(self.nclasses,activation=ks.activations.softmax()))

        #optimizer
        opt = ks.optimizers.RMSprop(lr=0.0001)

        ## Model compile
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        print(model.summary())

        return model


    # method for MLP model
    def get_mlp_model(self):
        np.random.seed(7)

        model = ks.Sequential()

        model.add(ks.layers.Dense(50, activation='relu', input_shape=(self.nfeatures,)))
        model.add(ks.layers.Dense(50, activation='relu'))
        model.add(ks.layers.Dense(50, activation='relu'))
        model.add(ks.layers.Dense(50, activation='relu'))
        model.add(ks.layers.Dense(50, activation='relu'))
        #model.add(tf.keras.layers.Dense(100, activation='relu'))
        model.add(ks.layers.Dense(self.nclasses, activation='softmax'))

        ## optimizer
        opt = ks.optimizers.RMSprop(lr=0.001)

        ## Model compile
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        print(model.summary())
        return model
