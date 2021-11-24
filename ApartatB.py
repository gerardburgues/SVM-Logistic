from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.preprocessing import LabelEncoder
import numpy as np

from sklearn.linear_model import LogisticRegression

import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score


def readDataset():
    return pd.read_csv('Iris.csv')


def defineData(df):
    x = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
    y = df['Species']
    return x, y


def encodeData(y):
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    return y


def LogisticRegression_Model():
    lr_model = LogisticRegression()
    lr_model.fit(x_train, y_train)
    lr_predict = lr_model.predict(x_test)
    print('Logistic Regression - Accuracy: ', accuracy_score(lr_predict, y_test))


def SVM_Model():
    svm_model = svm.SVC(kernel='linear')
    svm_model.fit(x_train, y_train)
    svc_predict = svm_model.predict(x_test)
    print('SVM - Accuracy: ', accuracy_score(svc_predict, y_test))


def RandomForest_Model():
    rfc_model = RandomForestClassifier(max_depth=3)
    rfc_model.fit(x_train, y_train)
    rfc_predict = rfc_model.predict(x_test)
    print('Random Forest - Accuracy: ', accuracy_score(rfc_predict, y_test))


def KNN_Model():
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(x_train, y_train)
    knn_predict = knn_model.predict(x_test)
    print('knn - Accuracy', accuracy_score(knn_predict, y_test))


if __name__ == "__main__":
    df = readDataset()
    x, y = defineData(df)
    y = encodeData(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)
    LogisticRegression_Model()
    SVM_Model()
    RandomForest_Model()

    KNN_Model()
