from nltk.tokenize import TweetTokenizer
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN
from MulticoreTSNE import MulticoreTSNE as TSNE

import pickle
import pandas
import numpy as np
import matplotlib.pyplot as plt
from lib.custFunctions import adjVector, HashtagVector, atVector

# Load data
df = pandas.read_csv('Subtask B/donaldTrumpTweets', sep='\t', encoding='latin1')

#Parse out not available tweets 
availableTweets = df.loc[df['Tweet'] != 'Not Available']
train, test = train_test_split(availableTweets, test_size=0.99)

print('adj vector')
adj = adjVector(train['Tweet'])
adj = adj.toarray()
print('hashtag vector')
hashTag = HashtagVector(train['Tweet'])
hashTag = hashTag.toarray()

print('at vector')
atvec = atVector(train['Tweet'])
atvec = atvec.toarray()
print('combining vector')

X = []
for x1,x2 in zip(atvec,adj):
    tmp = np.append(x1,x2)
    X.append(tmp)
X = np.array(X)

print(X)
print(X.shape)
# t-SNE embedding of the digits dataset
print("Computing t-SNE embedding")
tsne = TSNE(n_jobs=-1,verbose=1)
X_tsne = tsne.fit_transform(X)

with open('tsne.pickle', 'wb') as handle:
        pickle.dump(X_tsne, handle, protocol=pickle.HIGHEST_PROTOCOL)
