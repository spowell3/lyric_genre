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
	



