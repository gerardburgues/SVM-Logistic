import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
import pandas as pd
# import some data to play with
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, precision_recall_curve, average_precision_score, roc_curve, auc



def dropData(data):






    return data.drop(["Date","Evaporation","Sunshine", "Cloud9am", "Cloud3pm"] , axis=1)

def transformData(data):
    Location  = pd.get_dummies(data["Location"], prefix="loc_")

    WindGustDir = pd.get_dummies(data["WindGustDir"], prefix="WGD_")

    WindDir9am = pd.get_dummies(data["WindDir9am"], prefix="WG9a_")
    WindDir3pm = pd.get_dummies(data["WindDir3pm"], prefix="WG3p_")
    RainToday = pd.get_dummies(data["RainToday"], prefix="RT_")
    RainTomorrow = pd.get_dummies(data["RainTomorrow"], prefix="RTm_")

    data = data.join(Location)
    data = data.join(WindGustDir)
    data = data.join(WindDir9am)
    data = data.join(WindDir3pm)
    data = data.join(RainToday)
    data = data.join(RainTomorrow)
    data = data.drop(["Location","WindGustDir","WindDir9am","WindDir3pm","RainToday","RainTomorrow"], axis= 1)
    mean = []
    print(type(data))
    df = pd.DataFrame(data)
    for i in df.columns:
            df[i].fillna(value=data[i].mean(), inplace=True)
   # print(data)
    return data
def SplitData(data):
    dataframe = data.to_numpy()
    X = dataframe[:, 1:-1]
    y = dataframe[:, -1]
    return X,y

def SVMLogistic(X,y):


    particions = [0.5, 0.7, 0.8]
    fig, sub = plt.subplots(1, 2, figsize=(16, 6))
    sub[0].scatter(X[:, 0], y, c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    sub[1].scatter(X[:, 1], y, c=y, cmap=plt.cm.coolwarm, edgecolors='k')

    for part in particions:
        x_t, x_v, y_t, y_v = train_test_split(X, y, train_size=part)

        # Creem el regresor logístic
        logireg = LogisticRegression(C=2.0, fit_intercept=True, penalty='l2', tol=0.001, max_iter=1000)

        # l'entrenem
        logireg.fit(x_t, y_t)

        print("Correct classification Logistic ", part, "% of the data: ", logireg.score(x_v, y_v))

        print("We are here 0")
        # Creem el regresor logístic
        svc = svm.SVC(C=10.0, kernel='linear', gamma=0.9, probability=True,max_iter=100)

        # l'entrenem
        print("We are here 1")
        print(len(x_t))
        svc.fit(x_t, y_t)
        print("We are here 2")
        probs = svc.predict_proba(x_v)
        print("We are here 3")
        print("Correct classification SVM      ", part, "% of the data: ", svc.score(x_v, y_v))

    ROCFunction(y_v, probs, 3)


def NormalizeData(data):

    """We normalize data to have the values between 0 and 1 """

    min_max = MinMaxScaler()
    dataNormalize = min_max.fit_transform(data)

    dataNormalize =  pd.DataFrame(dataNormalize)

    return dataNormalize

def ROCFunction(y_v,probs, n_classes):


    # Compute Precision-Recall and plot curve
    precision = {}
    recall = {}
    average_precision = {}
    plt.figure()
    for i in range(n_classes):
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
if __name__ == "__main__":
    data = pd.read_csv('weatherAUS.csv')
    data = dropData(data)
    data = transformData(data)
    data = NormalizeData(data)
    X,y = SplitData(data)
    print("esto es X " ,X)
    print("esto es Y " ,y)
    SVMLogistic(X,y)



"""
dataframe = data.to_numpy()

X = dataframe[:,:-1]
y = dataframe[:, -1]

print("esto es X ", X)
print("esto es Y ", y)
"""
