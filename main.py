import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import svm, datasets, metrics
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import preprocessing
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, precision_recall_curve, average_precision_score, roc_curve, auc
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from scipy.special import expit, logit

import time
from DataAnalysis import DataAnalysis
import missingno as msno


def dropData(data):
    return data.drop(["Date"], axis=1)


def HotEncoderFunc(data):
    Location = pd.get_dummies(data["Location"], prefix="loc_")

    WindDir9am = pd.get_dummies(data["WindDir9am"], prefix="WG9a_")
    WindDir3pm = pd.get_dummies(data["WindDir3pm"], prefix="WG3p_")
    WindGustDir = pd.get_dummies(data["WindGustDir"], prefix="WGD_")

    data = data.join(Location)

    data = data.join(WindDir9am)
    data = data.join(WindDir3pm)
    data = data.join(WindGustDir)

    data = data.drop(["Location", "WindDir9am", "WindDir3pm", "WindGustDir"], axis=1)

    return data


def TransformingData(data):
    data['RainTomorrow'] = data['RainTomorrow'].map({'Yes': 1, 'No': 0})
    data['RainToday'] = data['RainToday'].map({'Yes': 1, 'No': 0})
    return data


def Encoder(data):
    le = preprocessing.LabelEncoder()
    data['Location'] = le.fit_transform(data['Location'])
    data['WindDir9am'] = le.fit_transform(data['WindDir9am'])
    data['WindDir3pm'] = le.fit_transform(data['WindDir3pm'])
    data['WindGustDir'] = le.fit_transform(data['WindGustDir'])
    return data


def SplitData(data):
    dataframe = data.to_numpy()
    X = dataframe[:, 1:-1]
    y = dataframe[:, -1]
    return X, y


def LogisticCrossVal(X, y):
    x_t, x_v, y_t, y_v = train_test_split(X, y, test_size=0.3)

    k = 5
    kf = KFold(n_splits=k, random_state=None)
    model = LogisticRegression(C=1.5, fit_intercept=True, penalty='l2', tol=0.001, max_iter=1000)

    acc_score = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        pred_values = model.predict(X_test)

        acc = accuracy_score(pred_values, y_test)
        acc_score.append(acc)

    avg_acc_score = sum(acc_score) / k

    print('accuracy of each fold - {}'.format(acc_score))
    print('Avg accuracy : {}'.format(avg_acc_score))


def Logistic(X, y):
    particions = [0.5, 0.7, 0.8]

    # for part in particions:
    x_t, x_v, y_t, y_v = train_test_split(X, y, test_size=0.3)

    # Creem el regresor logístic
    #logireg = LogisticRegression(C=2.0, fit_intercept=True, penalty='l2', tol=0.001, max_iter=1000)
    logireg = LogisticRegression()
    # l'entrenem
    logireg.fit(x_t, y_t)

    print("Correct classification Logistic ", 0.2, "% of the data: ", logireg.score(x_v, y_v))
    print(len(y))
    print(len(X))

    plt.scatter(X[:, 0].ravel(), y, color="black", zorder=20)
    loss = expit(x_v * logireg.coef_ + logireg.intercept_)
    print(loss)
    print(x_v[:, 0])
    plt.plot(x_v[: ,0], loss, color="red", linewidth=3)

    plt.ylim(0, 1)
    plt.xlim(0, 1)

    plt.ylabel('y')
    plt.xlabel('x')
    plt.show()
    # ROCFunction(y_v, probs, 3)


def SVM(X, y):
    x_t, x_v, y_t, y_v = train_test_split(X, y, test_size=0.2)

    print("We are here 0")
    # Creem el regresor logístic
    svc = svm.SVC(C=5.0, kernel='rbf', gamma=0.9, probability=True, max_iter=500)

    # l'entrenem
    print("We are here 1")
    print(len(x_t))
    svc.fit(x_t, y_t)
    print("We are here 2")
    probs = svc.predict_proba(x_v)
    print("We are here 3")
    print("Correct classification SVM      ", 0.2, "% of the data: ", svc.score(x_v, y_v))


def SVMCrossVal(X, y):
    x_t, x_v, y_t, y_v = train_test_split(X, y, test_size=0.2)

    k = 5
    kf = KFold(n_splits=k, random_state=None)
    model = svm.SVC(C=10.0, kernel='linear', gamma=0.9, probability=True, max_iter=500)

    acc_score = []
    probs_scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        pred_values = model.predict(X_test)

        acc = accuracy_score(pred_values, y_test)
        acc_score.append(acc)

    avg_acc_score = sum(acc_score) / k

    print('accuracy of each fold - {}'.format(acc_score))
    print('Avg accuracy : {}'.format(avg_acc_score))


def NullData(data):
    """

    For each parameter we are replacing de nans with the mean
    """
    data['MinTemp'] = data['MinTemp'].fillna(data['MinTemp'].mean())
    data['MaxTemp'] = data['MinTemp'].fillna(data['MaxTemp'].mean())
    data['Rainfall'] = data['Rainfall'].fillna(data['Rainfall'].mean())
    data['Evaporation'] = data['Evaporation'].fillna(data['Evaporation'].mean())
    data['Sunshine'] = data['Sunshine'].fillna(data['Sunshine'].mean())
    data['WindGustSpeed'] = data['WindGustSpeed'].fillna(data['WindGustSpeed'].mean())
    data['WindSpeed9am'] = data['WindSpeed9am'].fillna(data['WindSpeed9am'].mean())
    data['WindSpeed3pm'] = data['WindSpeed3pm'].fillna(data['WindSpeed3pm'].mean())
    data['Humidity9am'] = data['Humidity9am'].fillna(data['Humidity9am'].mean())
    data['Humidity3pm'] = data['Humidity3pm'].fillna(data['Humidity3pm'].mean())
    data['Pressure9am'] = data['Pressure9am'].fillna(data['Pressure9am'].mean())
    data['Pressure3pm'] = data['Pressure3pm'].fillna(data['Pressure3pm'].mean())
    data['Cloud9am'] = data['Cloud9am'].fillna(data['Cloud9am'].mean())
    data['Cloud3pm'] = data['Cloud3pm'].fillna(data['Cloud3pm'].mean())
    data['Temp9am'] = data['Temp9am'].fillna(data['Temp9am'].mean())
    data['Temp3pm'] = data['Temp3pm'].fillna(data['Temp3pm'].mean())

    """
    Here we are replacing the nans with the mode 
    """
    data['RainToday'] = data['RainToday'].fillna(data['RainToday'].mode()[0])
    data['RainTomorrow'] = data['RainTomorrow'].fillna(data['RainTomorrow'].mode()[0])
    data['WindDir9am'] = data['WindDir9am'].fillna(data['WindDir9am'].mode()[0])
    data['WindGustDir'] = data['WindGustDir'].fillna(data['WindGustDir'].mode()[0])
    data['WindDir3pm'] = data['WindDir3pm'].fillna(data['WindDir3pm'].mode()[0])

    return data


def NormalizeData(data):
    """We normalize data to have the values between 0 and 1 """

    min_max = MinMaxScaler()
    dataNormalize = min_max.fit_transform(data)
    dataNormalize = pd.DataFrame(dataNormalize)

    return dataNormalize


def ROCFunction(y_v, probs, n_classes=3):
    # Compute Precision-Recall and plot curve
    precision = {}
    recall = {}
    average_precision = {}
    plt.figure()
    for i in range(n_classes):
        print(y_v)
        print(probs[:, i])
        precision[i], recall[i], _ = precision_recall_curve(y_v == i, probs[:, i])
        average_precision[i] = average_precision_score(y_v == i, probs[:, i])

        plt.plot(recall[i], precision[i],
                 label='Precision-recall curve of class {0} (area = {1:0.2f})'
                       ''.format(i, average_precision[i]))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(loc="upper right")

    # Compute ROC curve and ROC area for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_v == i, probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    # Plot ROC curve
    plt.figure()
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})' ''.format(i, roc_auc[i]))
    plt.legend()
    plt.show()


def PCAFunc(data):
    pca = PCA(n_components=2)
    pca.fit(data)
    print(" explained variance ration", pca.explained_variance_ratio_, "\n")
    print("singular", pca.singular_values_)


def NearestNeighbour(X, y):
    """X_train, X_test, y_train, y_test = train_test_split(X, y,stratify = y, test_size = 0.7, random_state = 42)
    nca = NeighborhoodComponentsAnalysis(random_state=42)
    knn = KNeighborsClassifier(n_neighbors=3)
    nca_pipe = Pipeline([('nca', nca), ('knn', knn)])
    nca_pipe.fit(X_train, y_train)
    print(nca_pipe.score(X_test, y_test))"""

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    knn = KNeighborsClassifier(n_neighbors=2)

    # Train the model using the training sets
    knn.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = knn.predict(X_test)
    print(y_pred)
    print("Accuracy:", metrics.f1_score(y_test, y_pred))


def NearestNeighbourCrossVal(X, y):
    k = 5
    kf = KFold(n_splits=k, random_state=None)
    model = KNeighborsClassifier(n_neighbors=20)

    acc_score = []
    print("type_ : ", type(X))
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        pred_values = model.predict(X_test)

        acc = accuracy_score(pred_values, y_test)
        acc_score.append(acc)

    avg_acc_score = sum(acc_score) / k

    print('accuracy of each fold - {}'.format(acc_score))
    print('Avg accuracy : {}'.format(avg_acc_score))


def NullsNumber(data):
    "The below graphs show that the number of missing values are high in: Sunshine, Evaporation, Cloud3pm and Cloud9am."
    msno.heatmap(data)
    plt.show()
    msno.bar(data, sort="ascending")
    plt.show()
    print((data.isnull().sum() / len(data)) * 100)


if __name__ == "__main__":
    tic = time.perf_counter()
    data = pd.read_csv('weatherAUS.csv')
    DataAnalysis(data)
    NullsNumber(data)
    data = TransformingData(data)
    data = NullData(data)
    data = dropData(data)
    data = Encoder(data)
    # data= HotEncoderFunc(data)
    # Showing HeatMap
    correlation = data.corr()
    plt.figure(figsize=(20, 12))
    plt.title('Correlació de correlació')
    ax = sns.heatmap(correlation, square=True, annot=True, fmt='.2f', linecolor='white')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=30)
    plt.show()
    data = NormalizeData(data)

    X, y = SplitData(data)
    # print(X)
    # print(y)
    # PCAFunc(data)
    # SVM(X,y)
    #Logistic(X, y)
    NearestNeighbour(X,y)
    # SVMCrossVal(X,y)
    # LogisticCrossVal(X,y)
    # NearestNeighbourCrossVal(X,y)
    toc = time.perf_counter()
    print(f"Downloaded the tutorial in {toc - tic:0.4f} seconds")
