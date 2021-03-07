# Cameron J. Calv
# ECE 612: Applied Machine Learning
# Project 3
# 
# All required files may be found in the same folder here. 
# I also did a lot of preparatory and exploratory work in a Jupyter 
#   notebook located in the 'ref' directory (***.ipynb)
import os
import pandas
import pickle
import keras
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Config
PLOT_ON = False
LOAD_PARTIAL = True
REF_DIR = "ref\\in"
MEDIA_DIR = "ref\\out"
MODEL_NAME = "model3.h5"
PIPELINE_NAME = "pipeline3.pkl"

#Load up the test data from the Pickle file
def load_data(b_load_partial_data):
    if not b_load_partial_data:
        DATA_NAME = "project3trainset.pkl"
        LABEL_NAME = "project3trainlabel.pkl"
        data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), REF_DIR, DATA_NAME)
        with open(data_path, "rb") as data_in:
            data_X = pandas.read_pickle(data_in)
        label_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), REF_DIR, LABEL_NAME)
        with open(label_path, "rb") as data_in:
            data_Y = pandas.read_pickle(data_in)
    else:
        LABEL_NAME = "project3trainlabel.pkl"
        label_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), REF_DIR, LABEL_NAME)
        with open(label_path, "rb") as data_in:
            data_Y = pandas.read_pickle(data_in)
        for i in range(4):
            data_name = "project3trainset_0"+str(i+1)+".pkl"
            data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), REF_DIR, data_name)
            if i == 0:
                with open(data_path, "rb") as data_in:
                    data_X = pandas.read_pickle(data_in)
            else:
                with open(data_path, "rb") as data_in:
                    data_X = np.concatenate((data_X, pandas.read_pickle(data_in)))
    return [data_X, data_Y]

if __name__ == "__main__":
    #Load Complete Data
    [data_features, data_labels] = load_data(LOAD_PARTIAL)

    #Split data into sets
    new_shape = list(data_features.shape)
    new_shape.insert(3, 1)
    new_shape = tuple(new_shape)
    reshape_data = data_features.reshape(new_shape)

    data_train, data_val, label_train, label_val = train_test_split(reshape_data, data_labels, test_size=0.2)
    data_train_2, data_test, label_train_2, label_test = train_test_split(reshape_data, data_labels, test_size=0.1)

    #Define our model
    #First the ResidualUnit (from the book)
    class ResidualUnit(keras.layers.Layer):
        def __init__(self, filters, strides=1, activation="relu", **kwargs):
            super().__init__(**kwargs)
            self.activation = keras.activations.get(activation)
            self.main_layers = [
                keras.layers.Conv2D(filters, 3, strides=strides,
                    padding="same", use_bias=False),
                keras.layers.BatchNormalization(),
                self.activation,
                keras.layers.Conv2D(filters, 3, strides=1,
                    padding="same", use_bias=False),
                keras.layers.BatchNormalization()]
            self.skip_layers = []
            if strides > 1:
                self.skip_layers = [
                    keras.layers.Conv2D(filters, 1, strides=strides,
                    padding="same", use_bias=False),
                    keras.layers.BatchNormalization()]

        def call(self, inputs):
            Z = inputs
            for layer in self.main_layers:
                Z = layer(Z)
            skip_Z = inputs
            for layer in self.skip_layers:
                skip_Z = layer(skip_Z)
            return self.activation(Z + skip_Z)

        #Self-implemented for saving
        def get_config(self):
            base_config = super().get_config()
            return {**base_config, "activation": self.activation, "main_layers": self.main_layers, "skip_layers": self.skip_layers}


    #The rest of the architecture
    model_res = keras.models.Sequential()
    model_res.add(keras.layers.Conv2D(16, 7, strides=2, input_shape=(28, 28, 1),
        padding="same", use_bias=False))
    model_res.add(keras.layers.BatchNormalization())
    model_res.add(keras.layers.Activation("relu"))
    model_res.add(keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same"))
    for i in range(10):
        model_res.add(ResidualUnit(16, strides=1))
    model_res.add(keras.layers.GlobalAvgPool2D())
    model_res.add(keras.layers.Flatten())
    model_res.add(keras.layers.Dense(10, activation="softmax"))

    #Compile and fit
    model_res.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
    history = model_res.fit(data_train, label_train, epochs=10, validation_data=(data_val, label_val))
    score = model_res.evaluate(data_test, label_test)

    #Score and validate
    print("Reduced Res CNN:\n====================")
    print("Test Accuracy Score: "+str(model_res.evaluate(data_train, label_train)))
    print("Validation Accuracy Score: "+str(model_res.evaluate(data_val, label_val)))

    #Save and output
    model_res.save(os.path.join(os.path.realpath(__file__),"..", MODEL_NAME), save_format='h5')
    #Please load model with (new_model = tf.keras.models.load_model('model.h5', custom_objects={'ResidualUnit': ResidualUnit}))