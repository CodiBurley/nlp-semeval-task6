from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import TweetTokenizer
from sklearn.model_selection import cross_val_score
from sklearn import svm

import pandas


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


traindf = pandas.read_csv('trainingData.txt', sep='\t', encoding='latin1')
print(traindf.head())

testdf = pandas.read_csv('subtaskA-testdata-gold.txt', sep='\t', encoding='latin1')

outdf = pandas.DataFrame(columns=['ID','Target','Tweet','Stance'])

for topic in listOfTopics(traindf):
    extract = traindf.loc[traindf['Target'] == topic]
    corpus = list(map(lambda x: x[:-6], list(extract['Tweet'])))
    Y = list(extract['Stance'])
    ngram_vectorizer = CountVectorizer(ngram_range=(1, 1), analyzer='word', min_df=3, tokenizer=lambda x: custToken(x))
    X = ngram_vectorizer.fit_transform(corpus).toarray()

    clf = svm.SVC(kernel='linear')
    clf.fit(X,Y)

    testExtract = testdf.loc[testdf['Target'] == topic]
    testCorpus = list(map(lambda x: x[:-6], list(testExtract['Tweet'])))
    testX = ngram_vectorizer.transform(corpus).toarray()


    predictions = clf.predict(testX)

    ID = list(map(lambda x: str(x), list(testExtract['ID'])))
    s = zip(ID, list(testExtract['Target']), testCorpus, list(predictions))

    for x in s:
        outdf.loc[len(outdf)] = list(x)


outdf.set_index('ID')
outdf.to_csv('output.txt', sep='\t', index=False)
