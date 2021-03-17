# Cameron J. Calv
# ECE 612: Applied Machine Learning
# Project 4
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
import matplotlib.pyplot as plt
import matplotlib

#Config
PLOT_ON = False
LOAD_PARTIAL = True
REF_DIR = "ref\\in"
MEDIA_DIR = "ref\\out"
MODEL_NAME = "model4"
PIPELINE_NAME = "pipeline4.pkl"

#Load up the test data from the Pickle file
def load_data(b_load_partial_data):
    if not b_load_partial_data:
        DATA_NAME = "Xtrain.pkl"
        LABEL_NAME = "Ytrain.pkl"
        data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), REF_DIR, DATA_NAME)
        with open(data_path, "rb") as data_in:
            data_X = pandas.read_pickle(data_in)
        label_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), REF_DIR, LABEL_NAME)
        with open(label_path, "rb") as data_in:
            data_Y = pandas.read_pickle(data_in)
    else:
        for i in range(2):
            data_name = "project4cleanset_0"+str(i+1)+".pkl"
            data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), REF_DIR, data_name)
            if i == 0:
                with open(data_path, "rb") as data_in:
                    data_Y = pandas.read_pickle(data_in)
            else:
                with open(data_path, "rb") as data_in:
                    data_Y = np.concatenate((data_Y, pandas.read_pickle(data_in)))
        for i in range(2):
            data_name = "project4noiseset_0"+str(i+1)+".pkl"
            data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), REF_DIR, data_name)
            if i == 0:
                with open(data_path, "rb") as data_in:
                    data_X = pandas.read_pickle(data_in)
            else:
                with open(data_path, "rb") as data_in:
                    data_X = np.concatenate((data_X, pandas.read_pickle(data_in)))
    return [data_X, data_Y]

#For plotting purposes
def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instances[i:i+1].reshape(size,size) for i in range(instances.shape[0])]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = matplotlib.cm.binary, **options)
    plt.axis("off")

if __name__ == "__main__":
    #Load Complete Data
    [data_features, data_clean] = load_data(LOAD_PARTIAL)

    #Create the autoencoder
    conv_encoder = keras.models.Sequential([
        keras.layers.Reshape([28, 28, 1], input_shape=[28, 28]),
        keras.layers.Conv2D(8, kernel_size=3, padding="same", activation="selu"),
        keras.layers.MaxPool2D(pool_size=2),
        keras.layers.Conv2D(16, kernel_size=3, padding="same", activation="selu"),
        keras.layers.MaxPool2D(pool_size=2),
        # keras.layers.Conv2D(64, kernel_size=3, padding="same", activation="selu"),
        # keras.layers.MaxPool2D(pool_size=2)
    ])
    conv_decoder = keras.models.Sequential([
        keras.layers.Conv2DTranspose(16, kernel_size=2, strides=2,
            padding="valid",
            activation="selu",
            input_shape=[7, 7, 16]),
        # keras.layers.Conv2DTranspose(8, kernel_size=3, strides=2, padding="same",
        #     activation="selu"),# input_shape=[7, 7, 32],),
        keras.layers.Conv2DTranspose(1, kernel_size=3, strides=2, padding="same",
            activation="sigmoid"),
        keras.layers.Reshape([28, 28])
    ])
    conv_ae = keras.models.Sequential([conv_encoder, conv_decoder])
    conv_ae.compile(loss="binary_crossentropy",
        optimizer=keras.optimizers.SGD(lr=1.5))

    print(conv_encoder.summary())
    print(conv_decoder.summary())
    print(conv_ae.summary())

    #Train and predict
    history = conv_ae.fit(data_features, data_features, epochs=20, validation_data=[data_clean])
    codings = conv_ae.predict(data_features)

    #Give the results a plot
    plt.figure(figsize=(9,9))
    example_images = codings[:100]
    plot_digits(example_images, images_per_row=10)
    plt.show()

    #Save and output
    conv_ae.save(os.path.join(os.path.realpath(__file__),"..", MODEL_NAME+".h5"), save_format='h5')