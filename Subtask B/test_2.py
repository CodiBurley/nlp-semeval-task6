from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import train_test_split
import pandas
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import TweetTokenizer
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from matplotlib import offsetbox
import numpy as np
import matplotlib.pyplot as plt
import pandas
import pickle
import time
import os.path

# Load data
df = pandas.read_csv('donaldTrumpTweets', sep='\t', encoding='latin1')

#Parse out not available tweets 
availableTweets = df.loc[df['Tweet'] != 'Not Available']
print(availableTweets.head())

train, test = train_test_split(availableTweets, test_size=0.8)

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(train['Tweet'])

true_k = 3
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)

# t-SNE embedding of the digits dataset
print("Computing t-SNE embedding")
tsne = TSNE(n_components=2, init='pca', random_state=0, verbose=1)
t0 = time.time()
X_tsne = tsne.fit_transform(X.toarray())

plot_embedding(tsneFit)

plt.show()

def plot_embedding(X):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    plt.figure()

    x, y = X.T

    plt.scatter(x,y)