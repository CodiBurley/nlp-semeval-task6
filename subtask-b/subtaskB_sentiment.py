import pandas
import collections
import pandas
import subprocess
import numpy as np
import keras

from sklearn.model_selection import train_test_split
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from nltk.tokenize import TweetTokenizer
from features.tensed_pos_tag import get_x_data
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical

analyzer = SentimentIntensityAnalyzer()

# Load data
df = pandas.read_csv('data/train/donaldTrumpTweets', sep='\t', encoding='latin1')
# Parse out not available tweets
train = df.loc[df['Tweet'] != 'Not Available']
Y = []


# 0 is neutral
# 1 is against
# 2 is favor
def classify(score):
    pos = 0.1
    neg = -0.1
    if neg < score < pos:
        return 0
    if score < neg:
        return 1
    return 2


for idx, tweet in train.iterrows():
    Y.append(classify(analyzer.polarity_scores(tweet['Tweet'])['compound']))

counter = collections.Counter(Y)
print(counter)


def listOfTopics(dataframe):
    listOfTopics = []
    for row in traindf['Target']:
        if row not in listOfTopics:
            listOfTopics.append(row)
    return listOfTopics


def custToken(sent):
    tweet = TweetTokenizer().tokenize(sent)
    returnTweet = [w for w in tweet if '#' in w]
    return returnTweet


def convert_to_word(vec):
    num = vec.tolist().index(max(vec.tolist()))
    if num == 0:
        return"NONE"
    if num == 1:
        return "AGAINST"
    if num == 2:
        return "FAVOR"

traindf = train
testdf = pandas.read_csv('data/test/SemEval2016-Task6-subtaskB-testdata-gold.txt', sep='\t', encoding='latin1')
outdf = pandas.DataFrame(columns=['ID', 'Target', 'Tweet', 'Stance'])

print("Getting corpus...")
corpus = list(map(lambda x: x[:-6].lower(), list(traindf['Tweet'])))
print("Ngram Vectorizing...")
ngram_vectorizer = CountVectorizer(ngram_range=(1, 1), analyzer='word', min_df=5, tokenizer=lambda x: custToken(x))
print("Getting X Train Data...")
x_train = get_x_data(list(traindf['Tweet']), True)
print("Vectorizing...")
X = ngram_vectorizer.fit_transform(corpus).toarray()
X = np.concatenate((X, np.array(x_train)), axis=1)
# clf = svm.SVC(kernel='rbf', verbose=1)
# print("Training...")
# clf.fit(X, Y)

model = Sequential()
model.add(Dense(units=100, input_dim=X[0].shape[0]))  # layer
model.add(Activation('relu'))
model.add(Dense(units=64))
model.add(Activation('relu'))
model.add(Dense(units=32))
model.add(Activation('relu'))
model.add(Dense(units=3))  # layer
model.add(Activation('softmax'))  # layer

model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(lr=0.004),
              metrics=['accuracy']) # not layer

history = model.fit(X, to_categorical(np.array(Y), 3), validation_split=0.1, epochs=20, batch_size=512, verbose=2)

testCorpus = list(map(lambda x: x[:-6], list(testdf['Tweet'])))
x_test = get_x_data(list(testdf['Tweet']), True)
testX = ngram_vectorizer.transform(testCorpus).toarray()
testX = np.concatenate((testX, np.array(x_test)), axis=1)

predictions = list(map(convert_to_word, model.predict(testX)))

ID = list(map(lambda x: str(x), list(testdf['ID'])))
s = zip(ID, list(testdf['Target']), testCorpus, list(predictions))

for x in s:
    outdf.loc[len(outdf)] = list(x)


outdf.set_index('ID')
outdf.to_csv('output.txt', sep='\t', index=False)
subprocess.call(["perl", "eval.pl", "data/test/SemEval2016-Task6-subtaskB-testdata-gold.txt", "output.txt"])
