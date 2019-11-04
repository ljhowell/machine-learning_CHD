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


def scores(name, y_test, pred):
    accuracy = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)

    print("The accuracy score for {} is: {}%.".format(name, round(accuracy, 3) * 100))
    print("The f1 score for {} is: {}%.".format(name, round(f1, 3) * 100))
    print("The precision score for {} is: {}%.".format(name, round(precision, 3) * 100))
    print("The recall score for {} is: {}%.".format(name, round(recall, 3) * 100))

    return(accuracy, f1, precision, recall)


def model(classifier, name, X_train, X_test, y_train, y_test):
    classifier.fit(X_train, y_train)
    classifier_pred = classifier.predict(X_test)
    scores(name, y_test, classifier_pred)


print("Decision Tree")
model(DecisionTreeClassifier(), 'Decision Tree', X_train, X_test, y_train, y_test)

print("Random Forest")
model(RandomForestClassifier(n_estimators=100, bootstrap=True, max_features='sqrt'), 'Random Forest', X_train, X_test, y_train, y_test)
