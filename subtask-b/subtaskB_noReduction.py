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

testsize = 0.99
datasetprecent = 0.02

def outVect(lst,x,y,z):
    ret = []
    lbl = ['FAVOR', 'AGAINST', 'NONE']
    for x in test:
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

train, test = train_test_split(trump, test_size=testsize)
trump = train

test = pandas.read_csv('data/test/SemEval2016-Task6-subtaskB-testdata-gold.txt', \
    sep='\t', encoding='latin1')

outdf = pandas.DataFrame(columns=['ID', 'Target', 'Tweet', 'Stance'])

vec = vectorGen(trump)
print('Dbscanning')
dbscan = DBSCAN(eps=3.5, min_samples = \
    np.floor(len(trump)*datasetprecent), n_jobs = -1).fit(vec)
labels = dbscan.labels_
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters from training: %d' % n_clusters)

print('===================================================')
test_lbl = dbscan.fit_predict(vectorGen(test))

test_clusters = len(set(test_lbl)) - (1 if -1 in test_lbl else 0)
print('Estimated number of clusters from training: %d' % test_clusters)


ID = list(map(lambda x: str(x), list(test['ID'])))


lbl1 = outVect(test_lbl,0,1,2)
lbl2 = outVect(test_lbl,1,2,0)
lbl3 = outVect(test_lbl,2,0,1)



s = zip(ID, list(test['Target']), test['Tweet'], list(lbl1))

for x in s:
    outdf.loc[len(outdf)] = list(x)
outdf.set_index('ID')
outdf.to_csv('data/output1.txt', sep='\t', index=False)

# subprocess.call(["perl", "eval.pl", "data/test/subtaskB-testdata-gold.txt", \
#     "data/output.txt"])
