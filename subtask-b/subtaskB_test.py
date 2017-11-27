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

# Load data
df = pandas.read_csv('data/train/donaldTrumpTweets', sep='\t', encoding='latin1')

#Parse out not available tweets 
availableTweets = df.loc[df['Tweet'] != 'Not Available']
train, test = train_test_split(availableTweets, test_size=0.60)

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

db = DBSCAN(eps=3.5, min_samples = 2000, n_jobs = -1).fit(np.array(X_tsne))
labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)

output_file("DBSCAN.html")

colormap = {-1: 'black', 0: 'red', 1: 'green', 2: 'blue'}
colors = [colormap[x] for x in labels]

source = ColumnDataSource(data=dict(
    x=X_tsne[:,0],
    y=X_tsne[:,1],
    colors=colors,
    desc=list(df),
))

hover = HoverTool(tooltips=[
    ("index", "$index"),
    ("desc", "@desc"),
])

p = figure(tools=[hover,'pan','reset','wheel_zoom'], title="Mouse over the dots",plot_width=1000,plot_height=800)

p.circle('x', 'y', color='colors', fill_alpha=0.2, size=7, source=source)

show(p)
