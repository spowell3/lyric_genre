import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import *
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import classification_report,confusion_matrix
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


analyzer = TfidfVectorizer().build_analyzer()
stemmer = PorterStemmer()

#function that stems words
def stemmed_words(doc):   
    return (stemmer.stem(w) for w in analyzer(doc))

#Create vectorizer object
vectorizer = TfidfVectorizer(decode_error='ignore', stop_words='english', analyzer = stemmed_words)

#Read dataframe
lyrics = pd.read_csv('lyrics.csv')
# Subset the dataframe
lyrics_sub = lyrics[lyrics['genre'] != 'Not Available']
lyrics_sub = lyrics_sub[lyrics_sub['genre'] != 'Other']
genres = lyrics_sub['genre'].unique()


# DATA CLEANING
# Need to strip newline characters from lyrics

lyrics_sub['lyrics'] = lyrics_sub['lyrics'].str.replace(pat="\n", repl=' ')




#Transform lyrics using TFIDF, stemming, and removing stop words
word_vec = vectorizer.fit_transform(lyrics_sub['lyrics'].values.astype('U'))


X_train, X_test, y_train, y_test = train_test_split(word_vec, lyrics_sub['genre'], test_size=0.33)

#Create Complement Naive Bayes Model
clf = ComplementNB()
clf.fit(X_train, y_train)
pred = clf.predict(X_test)

#Evaluate Model
print (clf.score(X_test, y_test))
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
