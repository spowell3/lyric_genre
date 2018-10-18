import sklearn
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import *
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn import linear_model
from nltk.corpus import stopwords
import os
import pandas as pd
import numpy as np
import gensim
import nltk
import matplotlib.pyplot as plt
import re
import string
import csv
import pickle
import enchant
from nltk.stem.snowball import SnowballStemmer
from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

with open('fit_trans_even.pkl', 'rb') as f:
    word_vec = pickle.load(f)
print("done")

even = pd.read_csv('even.csv')

X_train, X_test, y_train, y_test = train_test_split(word_vec, even['genre'], test_size=0.20, stratify = even['genre'])


print("At CNB")
#Create Model
clf = ComplementNB()
clf.fit(X_train, y_train)
pred = clf.predict(X_test)

print("CNB Results")
#Score Model
print (clf.score(X_test, y_test))
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))

print("At XGB")
#Create Model
xgb = XGBClassifier()
xgb.fit(X_train, y_train)
pred = xgb.predict(X_test)

print("XGB RESULTS")
#Score Model
print(accuracy_score(y_test,pred))
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))


print ("At RF")
#Create Model
clf_4 = RandomForestClassifier()
clf_4.fit(X_train, y_train)
pred = clf_4.predict(X_test)

print("RF RESULTS")
#Score Model
print (clf_4.score(X_test, y_test))
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))


print("At Nearest Centroid")
#Create Model
clfc = NearestCentroid()
clfc.fit(X_train, y_train)
pred = clfc.predict(X_test)

print("NEAREST CENTROID RESULTS")
#Score Model
print (clfc.score(X_test, y_test))
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))


print("At SGD")
#Create Model
clfs = linear_model.SGDClassifier(max_iter=1000)
clfs.fit(X_train, y_train)
pred = clfs.predict(X_test)

print("SGD RESULTS")
#Score Model
print (clfs.score(X_test, y_test))
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))