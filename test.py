#Import
import os
import pandas
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

#Config
PLOT_ON = False
LOAD_PARTIAL = True
REF_DIR = "in"
MEDIA_DIR = "out"
MODEL_NAME = "model2.pkl"
PIPELINE_NAME = "pipeline2.pkl"

#Load up the test data from the Pickle file
def load_data(b_load_partial_data):
    if not b_load_partial_data:
        DATA_NAME = "mnist_X_train.pkl"
        LABEL_NAME = "mnist_y_train.pkl"
        data_path = os.path.join(REF_DIR, DATA_NAME)
        with open(data_path, "rb") as data_in:
            data_X = pandas.read_pickle(data_in)
        label_path = os.path.join(REF_DIR, LABEL_NAME)
        with open(label_path, "rb") as data_in:
            data_Y = pandas.read_pickle(data_in)
    else:
        LABEL_NAME = "mnist_y_train.pkl"
        label_path = os.path.join(REF_DIR, LABEL_NAME)
        with open(label_path, "rb") as data_in:
            data_Y = pandas.read_pickle(data_in)
        for i in range(4):
            data_name = "mnist_X_train_0"+str(i+1)+".pkl"
            data_path = os.path.join(REF_DIR, data_name)
            if (i == 0):
                with open(data_path, "rb") as data_in:
                    data_X = pandas.read_pickle(data_in)
            else:
                with open(data_path, "rb") as data_in:
                    data_X = np.concatenate((data_X, pandas.read_pickle(data_in)))
    return [data_X, data_Y]
[data_features, data_labels] = load_data(LOAD_PARTIAL)

#Split up the data
from sklearn.model_selection import train_test_split

data_train, data_val, label_train, label_val = train_test_split(data_features, data_labels, test_size=0.2)

# Support Vector Machine
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC 
from sklearn.linear_model import SGDClassifier
#Using GridSearchCV for optimization
from sklearn.model_selection import GridSearchCV

param_grid = [
        {'loss': ['hinge', 'huber'], 'penalty': ['l2', 'l1', 'elasticnet']},
        {'alpha': [1/1, 1/100, 1/10000], 'shuffle': [True, False], 'learning_rate': ['adaptive', 'optimal'],
        'early_stopping': [True, False]},
    ]

svm_clf = SGDClassifier()

grid_search = GridSearchCV(svm_clf, param_grid, cv=5,
    scoring='accuracy',
    return_train_score=True,
    n_jobs=-1)

grid_search.fit(data_features, data_labels)

#Let's see what it tells us the best parameters are
print(grid_search.best_params_)
print(grid_search.best_estimator_)