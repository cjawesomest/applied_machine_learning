#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 13:52:07 2021

@author: e-dag
"""
import os
import pickle
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import keras

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
        return {**base_config, "filters": 16, "strides": 1}
#%%%
# Import datasets

X = pd.read_pickle("project3trainset.pkl")
y = pd.read_pickle("project3trainlabel.pkl")
#%%%
from tensorflow import keras

#scaler = StandardScaler()
#X_train_scaled = scaler.fit_transform(X.astype(np.float64))

model = keras.models.load_model("model3.h5", custom_objects={'ResidualUnit': ResidualUnit})
X_scaled = X/255.0
#X_scaled = X
img_rows=X[0].shape[0]
img_cols=X[0].shape[1]

X_scaled=X_scaled.reshape(X_scaled.shape[0],img_rows,img_cols,1)
#%%%
# Predict on test set
y_hat = np.argmax(model.predict(X_scaled), axis=-1)

#%%
from sklearn.metrics import accuracy_score

print("classification accuracy is: ", accuracy_score(y, y_hat)*100)
