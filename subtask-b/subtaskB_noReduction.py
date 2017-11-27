from nltk.tokenize import TweetTokenizer
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN

import pickle
import pandas
import numpy as np
import matplotlib.pyplot as plt
from lib.custFunctions import adjVector, HashtagVector, atVector

def vectorGen(df):
    print('generating adj vector')
    adj = adjVector(df['Tweet'])
    adj = adj.toarray()
    print('generating hashtag vector')
    hashTag = HashtagVector(df['Tweet'])
    hashTag = hashTag.toarray()

    print('generating at vector')
    atvec = atVector(df['Tweet'])
    atvec = atvec.toarray()
    
    print('adj shape: ', adj.shape)
    print('hashtag shape: ', hashTag.shape)
    print('atvec shape: ', atvec.shape)


    print('combining vector')
    X = np.concatenate((adj, hashTag), axis=1)
    X = np.concatenate((X, atvec), axis=1)

    print(X.shape)
    return X


# Load data
trump = pandas.read_csv('data/train/donaldTrumpTweets', sep='\t', encoding='latin1')
trump = trump.loc[trump['Tweet'] != 'Not Available']

train, test = train_test_split(trump, test_size=0.95)
trump = train

test = pandas.read_csv('data/test/SemEval2016-Task6-subtaskB-testdata-gold.txt', \
    sep='\t', encoding='latin1')

outdf = pandas.DataFrame(columns=['ID', 'Target', 'Tweet', 'Stance'])

dbscan = DBSCAN(eps=3.5, min_samples = \
    np.floor(len(trump)*0.02), n_jobs = -1).fit(vectorGen(trump))
labels = dbscan.labels_
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters from training: %d' % n_clusters)


print('===================================================')
test = dbscan.fit_predict(vectorGen(test))

test_clusters = len(set(test)) - (1 if -1 in test else 0)
print('Estimated number of clusters from training: %d' % test_clusters)
