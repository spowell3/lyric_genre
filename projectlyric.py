# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import pandas as pd
import numpy as np
import gensim
import nltk
import matplotlib.pyplot as plt
import re
import string

# Set working directory
os.chdir('C:/Users/johnb/OneDrive/Documents/MSA/Fall 2/Text Mining/')

lyrics = pd.read_csv('lyrics.csv')
# Examine the dataframe
print(lyrics.head())

print(lyrics['genre'].unique())

print(np.sum(lyrics['genre'] == 'Not Available'))
print(np.sum(lyrics['genre'] == 'Other'))

print(lyrics['genre'].value_counts())
lyrics_sub = lyrics[lyrics['genre'] != 'Not Available']
lyrics_sub = lyrics_sub[lyrics_sub['genre'] != 'Other']
# Plot the frequency for the various genres
lyrics_sub['genre'].value_counts().plot(kind='bar')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.title('Frequency of Genres in Lyrics Dataset')
plt.show()

# DATA CLEANING
# Need to strip newline characters from lyrics

lyrics_sub['lyrics'] = lyrics_sub['lyrics'].str.replace(pat="\n", repl=' ')
print(lyrics_sub.head())
# Get the term vectors for the lyrics
test = lyrics_sub['lyrics'].head()
print(list(test))
punc = re.compile('[%s]' % re.escape(string.punctuation))
test_vec = []
for d in list(test):
	d = d.lower()
	d = punc.sub('',d)
	try:
		test_vec.append(nltk.word_tokenize(d))
	except:
		nltk.download('punkt')
		test_vec.append(nltk.word_tokenize(d))
	
for vec in test_vec:
	print(vec)
# This does it for all the songs so it takes a while
lyric_vec = []
for song in list(lyrics_sub['lyrics']):
	song = str(song).lower()
	song = punc.sub('',song)
	try:
		lyric_vec.append(nltk.word_tokenize(song))
	except:
		nltk.download('punkt')
		lyric_vec.append(nltk.word_tokenize(song))
	
print(lyric_vec[0])

try:
	stop_words = nltk.corpus.stopwords.words('english')
except:
	nltk.download('stopwords')
	stop_words = nltk.corpus.stopwords.words('english')

for i in range(0,len(lyric_vec)):
	term_list=[]
	for term in lyric_vec[i]:
		if term not in stop_words:
			term_list.append(term)
	lyric_vec[i] = term_list
	
print(lyric_vec[0])

porter = nltk.stem.porter.PorterStemmer()

for i in range(0, len(lyric_vec)):
	for j in range(0, len(lyric_vec[i])):
		lyric_vec[i][j] = porter.stem(lyric_vec[i][j])
		
print(lyric_vec[0])

dict = gensim.corpora.Dictionary(lyric_vec)

corp = []
for i in range(0, len(lyric_vec)):
    corp.append(dict.doc2bow(lyric_vec[i]))
    
tfidf_model = gensim.models.TfidfModel(corp)
tfidf = []
for i in range(0, len(corp)):
    tfidf.append(tfidf_model[corp[i]])
    
n = len(dict)
index = gensim.similarities.SparseMatrixSimilarity(tfidf_model[corp], num_features=n)


    
