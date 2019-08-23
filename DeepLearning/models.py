import tensorflow as tf
from numpy.random import seed


# Model

class DLmodel:
    """
    It has only one method and it is a static method which could be used to initialize the cnn model
    change code in the create_model method to change the architecture of the model
    """

    def __init__(self, nfeatures, nclasses):
        self.nfeatures = nfeatures
        self.nclasses = nclasses

    # method for 1D CNN model
    def get_1D_cnn_model(self):
        seed(7)
        tf.random.set_seed(3)

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Conv1D(10, kernel_size=1, activation='relu', input_shape=(1, self.nfeatures)))
        model.add(tf.keras.layers.Flatten())
<<<<<<< HEAD
        model.add(tf.keras.layers.Dense(50, activation='relu'))
        model.add(tf.keras.layers.Dense(50, activation='relu'))
        model.add(tf.keras.layers.Dense(50, activation='relu'))
        model.add(tf.keras.layers.Dense(50, activation='relu'))
=======
        model.add(tf.keras.layers.Dense(1000, activation='relu'))
        model.add(tf.keras.layers.Dense(500, activation='relu'))
>>>>>>> ff99370f6ded6a8cbe4c9b2375cc6ffd9ada95c6
        model.add(tf.keras.layers.Dense(50, activation='relu'))

        model.add(tf.keras.layers.Dense(self.nclasses, activation='softmax'))

        ## optimizer
        opt = tf.keras.optimizers.RMSprop(lr=0.0001)

        ## Model compile
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        print(model.summary())

        return model

    # method for MLP model
    def get_mlp_model(self):
        seed(7)
        tf.random.set_seed(3)

        model = tf.keras.Sequential()

        model.add(tf.keras.layers.Dense(300, activation='relu', input_shape=(self.nfeatures,)))
        model.add(tf.keras.layers.Dense(300, activation='relu'))
        model.add(tf.keras.layers.Dense(300, activation='relu'))
        model.add(tf.keras.layers.Dense(self.nclasses, activation='softmax'))

        ## optimizer
        opt = tf.keras.optimizers.RMSprop(lr=0.001)

        ## Model compile
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        print(model.summary())
        return model
