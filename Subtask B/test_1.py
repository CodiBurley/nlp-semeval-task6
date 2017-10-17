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


def custToken(sent):
    tweet = TweetTokenizer().tokenize(sent)
    returnTweet = [w for w in tweet if '#' in w]
    return returnTweet

def plotHashtag(sent):
    tweet = TweetTokenizer().tokenize(sent)
    returnTweet = [w for w in tweet if '#' in w]
    if returnTweet != []:
        return returnTweet[0]
    else:
        return 'na'

def plotColor(y):
    if y == 0:
        return 'r'
    elif y == 1:
        return 'g'
    else:
        return 'b'

# Load data
df = pandas.read_csv('donaldTrumpTweets', sep='\t', encoding='latin1')

#Parse out not available tweets 
availableTweets = df.loc[df['Tweet'] != 'Not Available']
print(availableTweets.head())

train, test = train_test_split(availableTweets, test_size=0.2)

#----------------------------------------------------------------------
# Scale and visualize the embedding vectors
def plot_embedding(X, p):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    plt.figure()

    x, y = X.T
    for i in range(5000):
    # #     str(custToken(train['Tweet'].iloc[i]))
    # #     # plt.text(X[i, 0], X[i, 1], str(i), fontdict={'size': 4})
        plt.scatter(X[i,0], X[i,1],color=plotColor(p[i]))
    #     print(str(X[i, 0]) + ' | ' + str(X[0, i]))


    plt.scatter(x,y)

#     if hasattr(offset    t0 = time.time()
# box, 'AnnotationBbox'):
#         # only print thumbnails with matplotlib > 1.0
#         shown_images = np.array([[1., 1.]])  # just something big
#         for i in range(digits.data.shape[0]):
#             dist = np.sum((X[i] - shown_images) ** 2, 1)
#             if np.min(dist) < 4e-3:
#                 # don't show points that are too close
#                 continue
#             shown_images = np.r_[shown_images, [X[i]]]
#             imagebox = offsetbox.AnnotationBbox(
#                 offsetbox.OffsetImage(digits.images[i], cmap=plt.c#----------------------------------------------------------------------m.gray_r),
#                 X[i])
#             ax.add_artist(imagebox)
#     plt.xticks([]), plt.yticks([])
#     if title is not None:
#         plt.title(title)


ngram_vectorizer = CountVectorizer(ngram_range=(1, 1), analyzer='word', stop_words='english', min_df=10, tokenizer=lambda x: custToken(x))
x = ngram_vectorizer.fit_transform(train['Tweet'])
model = KMeans(n_clusters=3,precompute_distances='auto',n_jobs=-1)

# fit and write a pickle if model hasn't already been fit

#fit
start = time.time()
print('fitting')
clf = model.fit(x)
end = time.time()
print('fitting complete')
print('time (s):',end-start)

#save
# pickle.dump(clf, handle, protocol=pickle.HIGHEST_PROTOCOL)




p=clf.predict(x)
print(p)

if not os.path.isfile('tsne.pickle'):

    # t-SNE embedding of the digits dataset
    print("Computing t-SNE embedding")
    tsne = TSNE(n_components=2, init='pca', random_state=0, verbose=1)
    t0 = time.time()
    X_tsne = tsne.fit_transform(x.toarray())

    with open('tsne.pickle', 'wb') as handle:
        pickle.dump(X_tsne, handle, protocol=pickle.HIGHEST_PROTOCOL)

tsneFit = None
with open('tsne.pickle', 'rb') as handle:
    tsneFit = pickle.load(handle)

plot_embedding(tsneFit, p)

plt.show()