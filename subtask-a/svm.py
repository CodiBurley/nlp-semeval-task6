from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from nltk.tokenize import TweetTokenizer
from features.tensed_pos_tag import get_xy_data

import pandas
import subprocess
import numpy as np


def listOfTopics(dataframe):
    listOfTopics = []
    for row in traindf['Target']:
        if row not in listOfTopics:
            listOfTopics.append(row)
    return listOfTopics

def yVec(l):
    ret = []
    for text in l:
        if text == 'AGAINST':
            ret.append(-1)
        elif text == 'FAVOR':
            ret.append(1)
        else:
            ret.append(0)
    return ret

def custToken(sent):
    tweet = TweetTokenizer().tokenize(sent)
    returnTweet = [w for w in tweet if '#' in w]
    return returnTweet


traindf = pandas.read_csv('data/train/trainingData.txt', sep='\t', encoding='latin1')
testdf = pandas.read_csv('data/test/subtaskA-testdata-gold.txt', sep='\t', encoding='latin1')
outdf = pandas.DataFrame(columns=['ID', 'Target', 'Tweet', 'Stance'])

for topic in listOfTopics(traindf):
    extract = traindf.loc[traindf['Target'] == topic]
    corpus = list(map(lambda x: x[:-6], list(extract['Tweet'])))
    Y = list(extract['Stance'])
    ngram_vectorizer = CountVectorizer(ngram_range=(1, 1), analyzer='word', min_df=2, tokenizer=lambda x: custToken(x))
    (x_train, _) = get_xy_data('data/train/trainingData.txt', topic, True)
    X = ngram_vectorizer.fit_transform(corpus).toarray()
    X = np.concatenate((X, np.array(x_train)), axis=1)
    clf = svm.SVC(kernel='rbf')
    clf.fit(X, Y)

    testExtract = testdf.loc[testdf['Target'] == topic]
    testCorpus = list(map(lambda x: x[:-6], list(testExtract['Tweet'])))
    (x_test, _) = get_xy_data('data/test/subtaskA-testdata-gold.txt', topic, True)
    testX = ngram_vectorizer.transform(testCorpus).toarray()
    testX = np.concatenate((testX, np.array(x_test)), axis=1)

    predictions = clf.predict(testX)

    ID = list(map(lambda x: str(x), list(testExtract['ID'])))
    s = zip(ID, list(testExtract['Target']), testCorpus, list(predictions))

    for x in s:
        outdf.loc[len(outdf)] = list(x)


outdf.set_index('ID')
outdf.to_csv('output.txt', sep='\t', index=False)
subprocess.call(["perl", "eval.pl", "data/test/subtaskA-testdata-gold.txt", "output.txt"])
