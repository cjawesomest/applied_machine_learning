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
import numpy as np

#Config
PLOT_ON = False
LOAD_PARTIAL = True
REF_DIR = "ref\\in"
MEDIA_DIR = "ref\\out"
MODEL_NAME = "model3.pkl"
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

        pass

    