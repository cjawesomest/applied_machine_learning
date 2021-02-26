# Cameron J. Calv
# ECE 612: Applied Machine Learning
# Project 2 
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
MODEL_NAME = "model2.pkl"
PIPELINE_NAME = "pipeline2.pkl"

#Load up the test data from the Pickle file
def load_data(b_load_partial_data):
    if not b_load_partial_data:
        DATA_NAME = "mnist_X_train.pkl"
        LABEL_NAME = "mnist_y_train.pkl"
        data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), REF_DIR, DATA_NAME)
        with open(data_path, "rb") as data_in:
            data_X = pandas.read_pickle(data_in)
        label_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), REF_DIR, LABEL_NAME)
        with open(label_path, "rb") as data_in:
            data_Y = pandas.read_pickle(data_in)
    else:
        LABEL_NAME = "mnist_y_train.pkl"
        label_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), REF_DIR, LABEL_NAME)
        with open(label_path, "rb") as data_in:
            data_Y = pandas.read_pickle(data_in)
        for i in range(4):
            data_name = "mnist_X_train_0"+str(i+1)+".pkl"
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

    #Split Up Data
    from sklearn.model_selection import train_test_split

    data_train, data_val, label_train, label_val = train_test_split(data_features, data_labels, test_size=0.2)

    #Model 1: SVM
    from sklearn.linear_model import SGDClassifier

    svm_model = SGDClassifier(loss='hinge', penalty='l2', 
        alpha=1, shuffle=True, learning_rate='adaptive',
        eta0=0.75, power_t=0.25, early_stopping=False)

    svm_model.fit(data_train, label_train)

    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import cross_val_predict

    print("SVM Alone:\n====================")
    print("Test Accuracy Score: " +str(
            accuracy_score(label_train, svm_model.predict(data_train), normalize=True, sample_weight=None)))
    print("Validation Accuracy Score: " +str(
            accuracy_score(label_val, svm_model.predict(data_val), normalize=True, sample_weight=None)))
    label_train_pred = cross_val_predict(svm_model, data_train, label_train, cv=5)
    print("Validation Accuracy Score with Cross-Validation: "+str(
            accuracy_score(label_train, label_train_pred, normalize=True, sample_weight=None)))

    #Model 2: Random Forest
    
    from sklearn.ensemble import RandomForestClassifier
    rf_model = RandomForestClassifier(n_estimators=1000, criterion='gini', 
        oob_score=True)

    rf_model.fit(data_train, label_train)
    
    print("Random Forest Alone:\n====================")
    print("Test Accuracy Score: " +str(
            accuracy_score(label_train, rf_model.predict(data_train), normalize=True, sample_weight=None)))
    print("Validation Accuracy Score: " +str(
            accuracy_score(label_val, rf_model.predict(data_val), normalize=True, sample_weight=None)))
    label_train_pred = cross_val_predict(rf_model, data_train, label_train, cv=5)
    print("Validation Accuracy Score with Cross-Validation: "+str(
            accuracy_score(label_train, label_train_pred, normalize=True, sample_weight=None)))
    

    #Model 3: Logiistic Regressor
    from sklearn.linear_model import LogisticRegression
    logr_model = LogisticRegression(C=0.0001, fit_intercept=False, multi_class='ovr',
        penalty='l1', solver='liblinear')

    logr_model.fit(data_train, label_train)

    print("Logistric Regression Alone:\n====================")
    print("Test Accuracy Score: " +str(
            accuracy_score(label_train, logr_model.predict(data_train), normalize=True, sample_weight=None)))
    print("Validation Accuracy Score: " +str(
            accuracy_score(label_val, logr_model.predict(data_val), normalize=True, sample_weight=None)))
    label_train_pred = cross_val_predict(logr_model, data_train, label_train, cv=5)
    print("Validation Accuracy Score with Cross-Validation: "+str(
            accuracy_score(label_train, label_train_pred, normalize=True, sample_weight=None)))

    #Model 4: Ensemble
    from sklearn.ensemble import VotingClassifier
    ensemble_model = VotingClassifier(
        estimators=[('svm', svm_model), ('rf', rf_model), ('logr', logr_model)],
        voting='hard')
    ensemble_model.fit(data_train, label_train)

    print("Ensemble:\n====================")
    print("Test Accuracy Score: " +str(
            accuracy_score(label_train, ensemble_model.predict(data_train), normalize=True, sample_weight=None)))
    print("Validation Accuracy Score: " +str(
            accuracy_score(label_val, ensemble_model.predict(data_val), normalize=True, sample_weight=None)))
    label_train_pred = cross_val_predict(ensemble_model, data_train, label_train, cv=5)
    print("Validation Accuracy Score with Cross-Validation: "+str(
            accuracy_score(label_train, label_train_pred, normalize=True, sample_weight=None)))

    # Save Models
    pickle.dump([svm_model, rf_model, logr_model, ensemble_model], open(os.path.join(os.path.realpath(__file__),"..", MODEL_NAME), "wb"))
