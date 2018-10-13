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


########################### Chelsey doing classification model ###################################################    
from collections import Counter
#from sklearn.naive_bayes import ComplementNB

#making pairs of lyrics and genres
lyrics_sub['clean_text'] = lyric_vec
pairs=[]
for i in range(0,len(lyrics_sub)):
    pairs.append((lyrics_sub.iloc[i,6], lyrics_sub.iloc[i,4]))
#lyrics are in the 6th column and genre is in the 4th column
np.random.shuffle(pairs)

#getting all the words to poplulate 2000 most frequenct for word_features

flattened_lyric_vec = [val for sublist in lyric_vec for val in sublist]
counts =  Counter(flattened_lyric_vec)
print(counts)

common_words = counts.most_common(2000)
print(common_words)
word_features = [i[0] for i in common_words]
print(word_features)

#function returns features of a song
def lyrics_features(doc):
    lyrics_words = set(doc)
    features = {}
    for word in word_features:
        features['contains(%s)' %word] = (word in lyrics_words)
    return features

#testing function
print(lyrics_features(pairs[0][0]))



#Classifying with a SMALL set of pairs
small_pairs = pairs[:1000]
featuresets = [(lyrics_features(l),g) for (l,g) in small_pairs] 
train_set, test_set = featuresets[500:], featuresets[:500]
classifier = nltk.NaiveBayesClassifier.train(train_set)

print(nltk.classify.accuracy(classifier, test_set))

classifier.show_most_informative_features(5)

#I RUN OUT OF MEMORY. :(
#Training and Testing classifier for lyric classificatoin - BIG 
featuresets = [(lyrics_features(l),g) for (l,g) in pairs] #THIS STEP IS WHERE I RUN OUT OF MEMORY!
train_set, test_set = featuresets[154370:], featuresets[:154370]
classifier = nltk.NaiveBayesClassifier.train(train_set)

print(nltk.classify.accuracy(classifier.test_set))

classifier.show_most_informative_features(5)
    
    
############################################################################## 




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


    
