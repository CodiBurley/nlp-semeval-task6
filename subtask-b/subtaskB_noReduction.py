from nltk.tokenize import TweetTokenizer
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN

import pickle
import pandas
import numpy as np
import matplotlib.pyplot as plt
from lib.custFunctions import adjVector, HashtagVector, atVector

def data(df):
    print('adj vector')
    adj = adjVector(df['Tweet'])
    adj = adj.toarray()
    print('hashtag vector')
    hashTag = HashtagVector(df['Tweet'])
    hashTag = hashTag.toarray()

    print('at vector')
    atvec = atVector(df['Tweet'])
    atvec = atvec.toarray()
    print('combining vector')
    X = []
    for x1,x2 in zip(atvec,adj):
        tmp = np.append(x1,x2)
        X.append(tmp)
    return np.array(X)



# Load data
trump = pandas.read_csv('data/train/donaldTrumpTweets', sep='\t', encoding='latin1')
trump = trump.loc[df['Tweet'] != 'Not Available']

test = pandas.read_csv('data/test/SemEval2016-Task6-subtaskB-testdata-gold.txt', sep='\t', encoding='latin1')

outdf = pandas.DataFrame(columns=['ID', 'Target', 'Tweet', 'Stance'])


db = DBSCAN(eps=3.5, min_samples = 2000, n_jobs = -1).fit(np.array(data(trump)))
labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

