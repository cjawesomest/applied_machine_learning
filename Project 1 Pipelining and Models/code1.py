# Cameron J. Calv
# ECE 612: Applied Machine Learning
# Project 1 
# 
# All required files may be found in the same folder here. 
# I also did a lot of preparatory and exploratory work in a Jupyter 
#   notebook located in the 'ref' directory (currency_prediction.ipynb)

import os
import pandas
import pickle
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np

#Config
PLOT_ON = False
REF_DIR = "ref\\in"
MEDIA_DIR = "ref\\out"
MODEL_NAME = "model1.pkl"
PIPELINE_NAME = "pipeline1.pkl"

#Load up the test data from the Pickle file
def load_data():
    DATA_NAME = "appml-assignment1-dataset.pkl"
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), REF_DIR, DATA_NAME)
    with open(data_path, "rb") as data_in:
        data = pandas.read_pickle(data_in)
    return [data['X'], data['y']]

def find_top_n_correlations(match_string, correlation_matrix, num):
    search_dict = correlation_matrix[match_string]
    sorted_score_idxs = [scores for scores in reversed(sorted(range(len(search_dict)), key=lambda k:search_dict[k]))][0:]
    scores = []
    temp_scores = []
    etfs = []
    temp_etfs = []
    top_cursor = 0
    bottom_cursor = len(sorted_score_idxs)-1
    for i in range(num):
        if abs(search_dict[sorted_score_idxs[top_cursor]]) >= abs(search_dict[sorted_score_idxs[bottom_cursor]]):
            next_score = search_dict[sorted_score_idxs[top_cursor]]
            next_etf = search_dict.keys()[sorted_score_idxs[top_cursor]]
            top_cursor = top_cursor + 1
        else:
            next_score = search_dict[sorted_score_idxs[bottom_cursor]]
            next_etf = search_dict.keys()[sorted_score_idxs[bottom_cursor]]
            bottom_cursor = bottom_cursor - 1
        temp_scores.append(next_score)
        temp_etfs.append(next_etf)
        if len(temp_scores) >= num:
            break
    final_sort_idxs = [scores for scores in reversed(sorted(range(len(temp_scores)), key=lambda k:abs(temp_scores[k])))]
    for idx in final_sort_idxs:
        scores.append(temp_scores[idx])
        etfs.append(temp_etfs[idx])
    return [scores, etfs]

class TopNTrimmer(BaseEstimator, TransformerMixin):
    def __init__(self, top_n):
        self.top_n = top_n
    def set_top_n(self, top_n):
        self.top_n = top_n
    def fit(self, X=None, y=None):
        return self
    def transform(self, X, y=None):
        #First get desired labels within data (dimension labels)
        from pandas.plotting import scatter_matrix
        DELIMETER = "-"
        etf_titles = set([x.split(DELIMETER)[0] for x in X.dtypes[1:].index])
        etf_stat = set([x.split(DELIMETER)[1] for x in X.dtypes[1:].index])

        open_attributes = []
        high_attributes = []
        low_attributes = []
        close_attributes = []

        for stat in etf_stat:
            for etf in etf_titles:
                if stat == 'open':
                    open_attributes.append(etf+DELIMETER+stat)
                elif stat == 'high':
                    high_attributes.append(etf+DELIMETER+stat)
                elif stat == 'low':
                    low_attributes.append(etf+DELIMETER+stat)
                elif stat == 'close':
                    close_attributes.append(etf+DELIMETER+stat)
                else:
                    pass
        #Determine correlations
        open_corr = data_features[open_attributes].corr()
        high_corr = data_features[high_attributes].corr()
        low_corr = data_features[low_attributes].corr()
        close_corr = data_features[close_attributes].corr()
        #Get the top correlators for open, high, low, and close
        [top_scores_open, top_etfs_open] = find_top_n_correlations('CAD-open', open_corr, self.top_n)
        [top_scores_high, top_etfs_high] = find_top_n_correlations('CAD-high', high_corr, self.top_n)
        [top_scores_low, top_etfs_low] = find_top_n_correlations('CAD-low', low_corr, self.top_n)
        [top_scores_close, top_etfs_close] = find_top_n_correlations('CAD-close', close_corr, self.top_n)
        #Trim accordingly
        data_features_trimmed = data_features[\
            top_etfs_close[0:self.top_n]+\
            top_etfs_high[0:self.top_n]+\
            top_etfs_low[0:self.top_n]+\
            top_etfs_open[0:self.top_n]]
        return data_features_trimmed

def train_model(model, X, y, max_wait=300):
    minimum_val_error = float("inf")
    best_train_length = None
    best_model = None
    since_last_decrease = 0
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=613)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_error = mean_squared_error(y_train[:m], y_train_predict)
        val_error = mean_squared_error(y_val, y_val_predict)
        if val_error < minimum_val_error:
            since_last_decrease = 0
            minimum_val_error = val_error
            best_train_length = m
            best_model = clone(model)
            best_model.fit(X_train[:m], y_train[:m])
        else:
            since_last_decrease = since_last_decrease + 1
        train_errors.append(train_error)
        val_errors.append(val_error)
        if since_last_decrease > max_wait:
            break
    return train_errors, val_errors, best_model, best_train_length

if __name__ == "__main__":
    #Load Data
    [data_features, data_labels] = load_data()

    #Build Pipeline
    #   -SelectTopN: select only the exchanges that are most correlated with CAD-high; reduces time with dimensionality reduction
    #   -SimpleImputer: replace any null or NaN values with the "mean" of the column
    #   -PolynomialFeatures: ensure that degree is size 1 before passing through to the LinearRegressor
    #   -StandardScaler: scale all feature values to be in line with the mean; for easier mathematics
    number_of_top_correlators = 4 #Best found was 4
    my_pipeline = Pipeline([
            ("top_n_trimmer", TopNTrimmer(number_of_top_correlators)),
            ("imputer", SimpleImputer(strategy="mean")), 
            ("poly_features", PolynomialFeatures(degree=1, include_bias=False)),
            ("scaler", StandardScaler()),
        ])
    my_pipeline.fit(data_features)

    #Transform Data
    data_final = my_pipeline.transform(data_features)

    #Train Model
    #   -Utilize Early Stopping: prevent model from adding too much data to the mix and stop after validation error stops decreasing
    model = LinearRegression(n_jobs=4)

    #LinearRegression fitting seems to work every Other time
    try:
        #Sweet, it worked the first time
        train, validate, final_model, m = train_model(model, data_final, data_labels)
    except np.linalg.LinAlgError:
        #In case it fails the first time, do it again
        train, validate, final_model, m = train_model(model, data_final, data_labels)

    #Statistics for Final Model
    #   -How low did we get our validation MSE?
    #   -How high did our training MSE get for the best model?
    mse_error = mean_squared_error(data_labels, final_model.predict(data_final))
    print("Model finished with MSE: "+str(mse_error))

    #Plot some "learning" curves
    fig = plt.figure(1, figsize=(12, 5))

    fig.add_subplot(121)
    plt.plot(train, "r-+", linewidth=2, label="train")
    plt.plot(validate, "b-", linewidth=3, label="val")
    plt.xlabel("Training Set Size")
    plt.ylabel("MSE")
    plt.legend()

    fig.add_subplot(122)
    plt.plot(train, "r-+", linewidth=2, label="train")
    plt.plot(validate, "b-", linewidth=3, label="val")
    plt.xlabel("Training Set Size")
    plt.ylabel("MSE")
    plt.ylim([0, min(validate) +(sum(train)/len(train))*2])
    plt.savefig(MEDIA_DIR+"\\learning_curves")
    plt.legend()
    if PLOT_ON:
        plt.show()

    #Save Model and Pipeline
    pickle.dump(final_model, open(MODEL_NAME, 'wb'))
    pickle.dump(my_pipeline, open(PIPELINE_NAME, 'wb'))