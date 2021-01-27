# Cameron J. Calv
# ECE 612: Applied Machine Learning
# Project 1 
import os
import pandas

def load_data():
    DATA_NAME = "appml-assignment1-dataset.pkl"
    REF_DIR = "ref"
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), REF_DIR, DATA_NAME)
    with open(data_path, "rb") as data_in:
        data = pandas.read_pickle(data_in)
    return [data['X'], data['y']]

if __name__ == "__main__":
    [data_features, data_labels] = load_data()
    pass