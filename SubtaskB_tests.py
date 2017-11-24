from nltk.tokenize import TweetTokenizer
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE

import pickle
import pandas
import numpy as np
import matplotlib.pyplot as plt
from lib.custFunctions import adjVector, HashtagVector


def plot_embedding(X):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    plt.figure()

    x, y = X.T

    plt.scatter(x,y)




# Load data
df = pandas.read_csv('Subtask B/donaldTrumpTweets', sep='\t', encoding='latin1')

#Parse out not available tweets 
availableTweets = df.loc[df['Tweet'] != 'Not Available']
print(availableTweets.head())


train, test = train_test_split(availableTweets, test_size=0.2)

adj = adjVector(train['Tweet'])
hashTag = HashtagVector(train['Tweet'])
hashTag = hashTag.toarray()

X = []
for x1,x2 in zip(adj,hashTag):
    tmp = np.append(x1,x2)
    X.append(tmp)
X = np.array(X)


# t-SNE embedding of the digits dataset
print("Computing t-SNE embedding")
tsne = TSNE(n_components=2, init='pca', random_state=0, verbose=1)
X_tsne = tsne.fit_transform(X)

with open('tsne.pickle', 'wb') as handle:
        pickle.dump(X_tsne, handle, protocol=pickle.HIGHEST_PROTOCOL)

colors = [(0,0,0,1),(0,0,1,1),(0,1,0,1),(1,0,0,1)]
ep = list(np.arange(4, 10, 0.2))
sp = list(range(1,150))

# x = 1
# for e in ep:
#     for s in sp:
db = DBSCAN(eps=6.4, min_samples = 76, n_jobs = -1).fit(np.array(X_tsne))
labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)
# for i, point in enumerate(X_tsne):
#     col = (0,1,1,1)
#     if labels[i] == -1:
#         col = colors[0]
#     elif labels[i] == 0:
#         col = colors[1]
#     elif labels[i] == 1:
#         col = colors[2]
#     elif labels[i] == 2:
#         col = colors[3]

#     plt.plot(point[0], point[1], 'o', markerfacecolor=col,
#              markeredgecolor='k', markersize=6)# for k, col in zip(unique_labels, colors):
# # print('Eps: ' + str(e) +  ' Samples: ' + str(s))
# # plt.savefig('eps_'+ str(e) + '_samples_' + str(s) + '_'+ str(x) + '.png')
# # plt.close()

# plt.show()