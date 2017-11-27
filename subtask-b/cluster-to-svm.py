from nltk.tokenize import TweetTokenizer
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn import svm

import pickle
import pandas
import subprocess
import random
import sys
import numpy as np
import matplotlib.pyplot as plt
from lib.custFunctions import adjVector, HashtagVector, atVector

trainingsize = 0.95
datasetprecent = 0.1
eps = 3.5

def outVect(lst):
    ret = []
    lbl = ['FAVOR', 'AGAINST', 'NONE']
    for x in lst:
        if x == -1:
            ret.append(random.choice(lbl))
        if x == 0:
            ret.append(lbl[0])
        if x == 1:
            ret.append(lbl[1])
        if x == 2:
            ret.append(lbl[2])
    return ret


def redistribute(lst,x=0,y=1,z=2):
    ret = []
    lbl = ['FAVOR', 'AGAINST', 'NONE']
    for p in lst:
        if p == 'FAVOR':
            ret.append(lbl[x])
        if p == 'AGAINST':
            ret.append(lbl[y])
        if p == 'NONE':
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

train, test = train_test_split(trump, test_size=trainingsize)
trump = train

vec = vectorGen(trump)

print("Computing t-SNE embedding")
tsne = TSNE(n_jobs=-1,verbose=1)
X_tsne = tsne.fit_transform(vec)

print('Dbscanning')
dbscan = DBSCAN(eps=eps, min_samples = \
    int(np.floor(len(trump)*datasetprecent)), n_jobs = -1).fit(X_tsne)
labels = dbscan.labels_
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print('Estimated number of clusters from training: %d' % n_clusters)

if n_clusters != 3:
    print('Wrong number of clusters quitting...')
    sys.exit()

clf = svm.SVC(kernel='rbf')
clf.fit(X_tsne, labels)


print('========================testing===========================')


test = pandas.read_csv('data/test/SemEval2016-Task6-subtaskB-testdata-gold.txt', \
    sep='\t', encoding='latin1')

test['Tweet'] = test['Tweet'].map(lambda x: x[:-6])

predictions = clf.predict(tsne.fit_transform(vectorGen(test)))

ID = list(map(lambda x: str(x), list(test['ID'])))


print('========================output1===========================')

lbl1 = outVect(predictions)

s = zip(ID, list(test['Target']), test['Tweet'], list(lbl1))

outdf = pandas.DataFrame(columns=['ID', 'Target', 'Tweet', 'Stance'])
for x in s:
    outdf.loc[len(outdf)] = list(x)

outdf.set_index('ID')
outdf.to_csv('data/output1.txt', sep='\t', index=False)

subprocess.call(["perl", "eval.pl", "data/test/SemEval2016-Task6-subtaskB-testdata-gold.txt", \
    "data/output1.txt"])

print('========================output2===========================')

lbl2 = redistribute(lbl1,1,2,0)

s = zip(ID, list(test['Target']), test['Tweet'], list(lbl2))

outdf = pandas.DataFrame(columns=['ID', 'Target', 'Tweet', 'Stance'])
for x in s:
    outdf.loc[len(outdf)] = list(x)

outdf.set_index('ID')
outdf.to_csv('data/output2.txt', sep='\t', index=False)

subprocess.call(["perl", "eval.pl", "data/test/SemEval2016-Task6-subtaskB-testdata-gold.txt", \
    "data/output2.txt"])
print('========================output3===========================')

lbl3 = redistribute(lbl1,2,0,1)

s = zip(ID, list(test['Target']), test['Tweet'], list(lbl3))

outdf = pandas.DataFrame(columns=['ID', 'Target', 'Tweet', 'Stance'])
for x in s:
    outdf.loc[len(outdf)] = list(x)

outdf.set_index('ID')
outdf.to_csv('data/output3.txt', sep='\t', index=False)
subprocess.call(["perl", "eval.pl", "data/test/SemEval2016-Task6-subtaskB-testdata-gold.txt", \
    "data/output3.txt"])