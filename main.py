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
import missingno as msno


def dropData(data):

    return data.drop(["Date","Evaporation","Sunshine", "Cloud9am", "Cloud3pm"] , axis=1)
def OrdinalEncoderFunc(data):
    """
    This function is an OrdinalEncoderFunction using sklearn
    Also Fill Nans with mean values
    """
    enc = OrdinalEncoder()
    enc.fit(data[["Location","WindGustDir","WindDir9am","WindDir3pm","RainToday","RainTomorrow"]])
    data[["Location","WindGustDir","WindDir9am","WindDir3pm","RainToday","RainTomorrow"]] = \
        enc.transform(data[["Location","WindGustDir","WindDir9am","WindDir3pm","RainToday","RainTomorrow"]])

    df = pd.DataFrame(data)
    for i in df.columns:
            df[i].fillna(value=data[i].mean(), inplace=True)
    return df
def OneHotEncoder(data):
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
    return df
def SplitData(data):
    dataframe = data.to_numpy()
    X = dataframe[:, 1:-1]
    y = dataframe[:, -1]
    return X,y
def SVMLogistic(X,y):


    particions = [0.5, 0.7, 0.8]

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
def StandarizeData(data):

    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)
    scaled = pd.DataFrame(scaled)
    return scaled
def NormalizeData(data):

    """We normalize data to have the values between 0 and 1 """

    min_max = MinMaxScaler()
    dataNormalize = min_max.fit_transform(data)

    dataNormalize =  pd.DataFrame(dataNormalize)

    return dataNormalize
def ROCFunction(y_v,probs, n_classes=3):


    # Compute Precision-Recall and plot curve
    precision = {}
    recall = {}
    average_precision = {}
    plt.figure()
    for i in range(n_classes):
        print(y_v)
        print(probs[:,i])
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
    pca  = PCA(n_components=2)
    pca.fit(data)
    print(" explained variance ration", pca.explained_variance_ratio_,"\n")
    print("singular" , pca.singular_values_)
def NearestNeighbour(X,y):
    """X_train, X_test, y_train, y_test = train_test_split(X, y,stratify = y, test_size = 0.7, random_state = 42)
    nca = NeighborhoodComponentsAnalysis(random_state=42)
    knn = KNeighborsClassifier(n_neighbors=3)
    nca_pipe = Pipeline([('nca', nca), ('knn', knn)])
    nca_pipe.fit(X_train, y_train)
    print(nca_pipe.score(X_test, y_test))"""

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

    knn = KNeighborsClassifier(n_neighbors=5)

    # Train the model using the training sets
    knn.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = knn.predict(X_test)
    print(y_pred)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
def NullsNumber(data):
    "The below graphs show that the number of missing values are high in: Sunshine, Evaporation, Cloud3pm and Cloud9am."
    msno.heatmap(data)
    plt.show()
    msno.bar(data, sort="ascending")
    plt.show()
    print((data.isnull().sum() / len(data)) * 100)


if __name__ == "__main__":

    data = pd.read_csv('weatherAUS.csv')
    print("Choose what type of Normalization you want: Standard or MinMax : \n")
    NullsNumber(data)
    NormType =  input()
    #print("How Nan affect Data: choose deleting rows (DEL) or NAn = mean value (MEAN) \n")
    #NanType = input()
    print("What type of Encoder do you want Ordinal or Hot : \n")
    Encoder = input()

    #Deleting rows we don't want
    data = dropData(data)
    if Encoder == "Hot":
        data = OneHotEncoder(data)
    else:
        data = OneHotEncoder(data)
        #data = OrdinalEncoderFunc(data)
        print("Not Ready")

    if NormType == "MinMax":
        data = NormalizeData(data)
    else:
        data = NormalizeData(data)
        print("StandarizeData() Does not work because does not go 0 to 1")
        #data = StandarizeData(data)

    X,y = SplitData(data)
    print(X)
    print(y)
    PCAFunc(data)
    #SVMLogistic(X,y)
    NearestNeighbour(X,y)



"""
dataframe = data.to_numpy()

X = dataframe[:,:-1]
y = dataframe[:, -1]

print("esto es X ", X)
print("esto es Y ", y)
"""
