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
# os.chdir('C:/Users/johnb/OneDrive/Documents/MSA/Fall 2/Text Mining/')
os.chdir('C:/Users/Steven/Documents/MSA/Analytics Foundations/Text/lyrics')

lyrics = pd.read_csv('lyrics.csv')
# Examine the dataframe
print(lyrics.head())

print(lyrics['genre'].unique())

print(np.sum(lyrics['genre'] == 'Not Available'))
print(np.sum(lyrics['genre'] == 'Other'))

print(lyrics['genre'].value_counts())
lyrics_sub = lyrics[lyrics['genre'] != 'Not Available']
lyrics_sub = lyrics_sub[lyrics_sub['genre'] != 'Other']
genres = lyrics_sub['genre'].unique()

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
#lyrics are in the 7th column and genre is in the 5th column
np.random.shuffle(pairs)
pairs[0]

#getting all the words to poplulate 2000 most frequenct for word_features
flattened_lyric_vec = [val for sublist in lyric_vec for val in sublist]
flattened_lyric_vec[88:120]
counts =  Counter(flattened_lyric_vec)
print(counts)

common_words = counts.most_common(2000)
print(common_words)
word_features = [i[0] for i in common_words]
print(word_features)

# NOTE: alternative attempt deviates here

##function returns features of a song
#def lyrics_features(doc):
#    lyrics_words = set(doc)
#    features = {}
#    for word in word_features:
#        features['contains(%s)' %word] = (word in lyrics_words)
#    return features
#
##testing function
#print(lyrics_features(pairs[0][0]))
#
##Defining featureset of SMALL set of pairs
#small_pairs = pairs[:1000]
#featuresets = [(lyrics_features(l),g) for (l,g) in small_pairs]
# 
## Classifying
#train_set, test_set = featuresets[500:], featuresets[:500]
#classifier = nltk.NaiveBayesClassifier.train(train_set)
#
#print(nltk.classify.accuracy(classifier, test_set))
#
#classifier.show_most_informative_features(5)
#
##I RUN OUT OF MEMORY. :(
##Training and Testing classifier for lyric classificatoin - BIG 
#featuresets = [(lyrics_features(l),g) for (l,g) in pairs] #THIS STEP IS WHERE I RUN OUT OF MEMORY!
#train_set, test_set = featuresets[154370:], featuresets[:154370]
#classifier = nltk.NaiveBayesClassifier.train(train_set)
#
#print(nltk.classify.accuracy(classifier.test_set))
#
#classifier.show_most_informative_features(5)
#     
############################################################################## 


###### ALTERNATIVE ATTEMPT ######
## feature function alternative
from collections import defaultdict
def get_features(song):
    features = defaultdict(bool)
    for word in song:
        features[word] = True
    return features
print(get_features(pairs[0][0]))

# Classifying on SMALL alternative set
from nltk.classify import apply_features

small_pairs = pairs[:1000]
train_set = apply_features(get_features, pairs[500:])
test_set = apply_features(get_features, pairs[:500])
classifier = nltk.NaiveBayesClassifier.train(train_set)

print(nltk.classify.accuracy(classifier, test_set))

classifier.show_most_informative_features(5)

# Assign a split to train and test
from sklearn.model_selection import train_test_split

# TODO attempt stratified sampling

np.random.seed(505) # I think this guarantees reproducible split for the function below.
train_pairs, test_pairs = train_test_split(pairs, test_size=0.2)
train_set = apply_features(get_features, train_pairs) #invisible in variable explorer...?
test_set = apply_features(get_features, test_pairs)
print(train_set[0][0])
print(train_pairs[5])

# Classifying on FULL alternative set
classifier = nltk.NaiveBayesClassifier.train(train_set)

print(nltk.classify.accuracy(classifier, test_set))

#getting list of individual predictions and actual results for confusion matrix
test_predictions=[]
test_actual=[]
for (featureset, genre) in test_set:
    #this is redundant, but I couldn't figure out another way to do this
    test_predictions.append(classifier.classify(dict(featureset)))
    test_actual.append(genre)

classifier.show_most_informative_features(5)

# Plotting a confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = nltk.ConfusionMatrix(test_predictions, test_actual) #not sure how to work with this object
cm2= confusion_matrix(test_predictions,test_actual) #using sklearn's tools
cm2norm= cm2.astype('float') / cm2.sum(axis=1)[:, np.newaxis] #normalizing to proportions
cm2_df=pd.DataFrame(cm2,columns=genres,index=genres)

fig, ax = plt.subplots(figsize=(8,8))
sns.heatmap(cm2_df,annot=True, ax=ax)
ax.set_title('Naive Bayes Confusion Matrix Heatmap')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# sanity check of distributions

ax=plt.axes()
# sort=False important, otherwise it'll scramble genre order. Not ideal for comparison
pd.Series(test_actual).value_counts(sort=False).plot(kind='bar')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.title('Frequency of Actual Genres')
plt.show()

ax=plt.axes()
pd.Series(test_predictions).value_counts(sort=False).plot(kind='bar')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.title('Frequency of Predicted Genres')
plt.show()



# TODO Ensure valid link b/w predictions and original metadata (ie artist, song, yr)

###################################


# Topic modeling

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


    
