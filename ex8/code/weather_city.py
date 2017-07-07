
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.color import lab2rgb
from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import sys
from sklearn.metrics import accuracy_score
from skimage import color
from sklearn import pipeline
from sklearn import preprocessing
from sklearn import svm
from sklearn.preprocessing import StandardScaler


def main():
    # filename1 = "monthly-data-labelled.csv"
    # filename2 = "monthly-data-unlabelled.csv"
    # filename3 = "labels.csv"

    filename1 = sys.argv[1]
    filename2 = sys.argv[2]
    filename3 = sys.argv[3]
    data = pd.read_csv(filename1)
    unlabelled_data = pd.read_csv(filename2)

    training_columns = data.columns.tolist()
    training_columns.remove("city")
    training_columns.remove("year")
    
    training_columns
    X_train,X_test,y_train,y_test = model_selection.train_test_split(data[training_columns].values,data["city"].values)

    svc_model = pipeline.make_pipeline(StandardScaler(),svm.SVC(kernel="linear", decision_function_shape="ovr"))
    svc_model.fit(X_train,y_train)
    
    predictions = svc_model.predict(unlabelled_data[training_columns].values)
    pd.Series(predictions).to_csv(filename3, index=False)

    Y_predicted_svc = svc_model.predict(X_test)
    print(accuracy_score(y_test, Y_predicted_svc))

if __name__ == '__main__':
    main()