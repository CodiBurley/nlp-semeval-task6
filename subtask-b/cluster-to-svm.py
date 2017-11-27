from nltk.tokenize import TweetTokenizer
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN

import pickle
import pandas
import subprocess
import random
import numpy as np
import matplotlib.pyplot as plt
from lib.custFunctions import adjVector, HashtagVector, atVector

trainingsize = 0.8
datasetprecent = 0.01
eps = 1

def outVect(lst,x,y,z):
    ret = []
    lbl = ['FAVOR', 'AGAINST', 'NONE']
    for x in lst:
        if x == -1:
            ret.append(random.choice(lbl))
        if x == 0:
            ret.append(lbl[x])
        if x == 1:
            ret.append(lbl[y])
        if x == 2:
            ret.append(lbl[z])
    return ret

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

vec = vectorGen(trump)
print('Dbscanning')
dbscan = DBSCAN(eps=eps, min_samples = \
    int(np.floor(len(trump)*datasetprecent)), n_jobs = -1).fit(vec)
labels = dbscan.labels_
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print('Estimated number of clusters from training: %d' % n_clusters)

