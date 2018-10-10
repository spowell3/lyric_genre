import os
import pandas as pd
import numpy as np
import gensim
import nltk
import matplotlib.pyplot as plt


os.chdir('C:/Users/johnb/OneDrive/Documents/MSA/Fall 2/Text Mining/')

lyrics = pd.read_csv('lyrics.csv')

print(lyrics.head())

print(lyrics['genre'].unique())

print(np.sum(lyrics['genre'] == 'Not Available'))
print(np.sum(lyrics['genre'] == 'Other'))

print(lyrics['genre'].value_counts())
lyrics_sub = lyrics[lyrics['genre'] != 'Not Available']
lyrics_sub = lyrics_sub[lyrics_sub['genre'] != 'Other']

lyrics_sub['genre'].value_counts().plot(kind='bar')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.title('Frequency of Genres in Lyrics Dataset')
plt.show()