from nltk.tokenize import TweetTokenizer
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN
from MulticoreTSNE import MulticoreTSNE as TSNE

import pickle
import pandas
import numpy as np
import matplotlib.pyplot as plt
from bokeh.plotting import figure, show, output_file
from bokeh.models import HoverTool, ColumnDataSource
from lib.custFunctions import adjVector, HashtagVector, atVector
import subprocess


def outVect(lst,x,y,z):
    ret = []
    lbl = ['FAVOR', 'AGAINST', 'NONE']
    for x in lst:
        if x == -1:
            ret.append(lbl[2])
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
df = pandas.read_csv('data/train/SemEval2016-Task6-subtaskB-testdata-gold.txt', sep='\t', encoding='latin1')

#Parse out not available tweets 
tweetDF = df.loc[df['Tweet'] != 'Not Available']
#train, test = train_test_split(availableTweets, test_size=0.60)

X = vectorGen(tweetDF)

print(X)
print(X.shape)
# t-SNE embedding of the digits dataset
print("Computing t-SNE embedding")
tsne = TSNE(n_jobs=-1,verbose=1)
X_tsne = tsne.fit_transform(X)

test_lbl = dbscan.fit_predict(X_tsne)

test_clusters = len(set(test_lbl)) - (1 if -1 in test_lbl else 0)
print('Estimated number of clusters from training: %d' % test_clusters)


ID = list(map(lambda x: str(x), list(test['ID'])))


lbl1 = outVect(test_lbl,0,1,2)
lbl2 = outVect(test_lbl,1,2,0)
lbl3 = outVect(test_lbl,2,0,1)


outdf = pandas.dataframe(columns=['id', 'target', 'tweet', 'stance'])
outdf1 = pandas.dataframe(columns=['id', 'target', 'tweet', 'stance'])
outdf2 = pandas.dataframe(columns=['id', 'target', 'tweet', 'stance'])

s = zip(ID, list(test['Target']), test['Tweet'], list(lbl1))
s1 = zip(ID, list(test['Target']), test['Tweet'], list(lbl2))
s2 = zip(ID, list(test['Target']), test['Tweet'], list(lbl3))

for x in s:
    outdf.loc[len(outdf)] = list(x)

outdf.set_index('ID')
outdf.to_csv('data/output1.txt', sep='\t', index=False)

subprocess.call(["perl", "eval.pl", "data/test/SemEval2016-Task6-subtaskB-testdata-gold.txt", \
    "data/output1.txt"])


for x in s1:
    outdf1.loc[len(outdf1)] = list(x)

outdf1.set_index('ID')
outdf1.to_csv('data/output2.txt', sep='\t', index=False)

subprocess.call(["perl", "eval.pl", "data/test/SemEval2016-Task6-subtaskB-testdata-gold.txt", \
    "data/output2.txt"])


for x in s2:
    outdf2.loc[len(outdf2)] = list(x)

outdf2.set_index('ID')
outdf2.to_csv('data/output3.txt', sep='\t', index=False)

subprocess.call(["perl", "eval.pl", "data/test/SemEval2016-Task6-subtaskB-testdata-gold.txt", \
    "data/output3.txt"])
