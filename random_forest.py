import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix

from preprocessing_ml import *

data = pd.read_csv("framingham.csv")

X_train, X_test, y_train, y_test = split_data(scale_data(drop_missing(chose_features(data))))


def scores(name, y_test, pred, v):
    accuracy = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)

    if v == 'y':
        print("The accuracy score for {} is: {}%.".format(name, round(accuracy, 3) * 100))
        print("The f1 score for {} is: {}%.".format(name, round(f1, 3) * 100))
        print("The precision score for {} is: {}%.".format(name, round(precision, 3) * 100))
        print("The recall score for {} is: {}%.".format(name, round(recall, 3) * 100))
        return(accuracy, f1, precision, recall)
    elif v == 'n':
        return(accuracy, f1, precision, recall)
    else:
        print("Error: please choose 'y' or 'n'.")


def model(classifier, name, X_train, X_test, y_train, y_test, v):
    classifier.fit(X_train, y_train)
    classifier_pred = classifier.predict(X_test)
    score = scores(name, y_test, classifier_pred, v)

    accuracy = score[0]
    f1 = score[1]
    precision = score[2]
    recall = score[3]

    return(accuracy, f1, precision, recall)


print("Decision Tree")
model(DecisionTreeClassifier(), 'Decision Tree', X_train, X_test, y_train, y_test, 'y')

print("Random Forest")
model(RandomForestClassifier(n_estimators=1000, bootstrap=True, max_features='sqrt'), 'Random Forest', X_train, X_test, y_train, y_test, 'y')
