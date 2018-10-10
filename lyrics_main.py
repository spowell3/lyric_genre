import gensim
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

def clean(dirty_words):
    # TODO Clean out labels of lyric sections (ie 'chorus', 'bridge')
    # TODO Clean out artist names from lyrics
    # TODO Replace special characters
    return(clean_words)

df = pd.read_csv("lyrics.csv", header=0) #TODO make sure lines are properly split
df['char_len'] = df['lyrics'].str.len()
df.describe(include = 'all')
dfsumm = df.groupby(['genre']).agg(['count', 'mean'])
dfsumm.columns = dfsumm.columns.map('_'.join) # removing multi-index
dfsumm = dfsumm.reset_index()
dfsumm['no_lyrics'] = dfsumm['index_count'] - dfsumm['char_len_count'] # calculating missing based on those without a character length

genre_map = {0: 'Country',
                1: 'Electronic',
                2: 'Folk',
                3: 'Hip-Hop',
                4: 'Indie',
                5: 'Jazz',
                6: 'Metal',
                7: 'NA',
                8: 'Other',
                9: 'Pop',
                10: 'R&B',
                11: 'Rock'}

# TODO get series of genres from dfsumm

df_lyr = df['lyrics'].apply(lambda x: clean(x)) # This line will likely kill memory on full dataset
df_gen= df['genre']


np.random.seed(505)
lyr_train, lyr_test, gen_train, gen_test = train_test_split(df_lyr, df_gen, test_size=0.2)


# Topic model within genres

# Identify clusters within genres

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=len(genre_map))
# kmeans.fit(___)


pass