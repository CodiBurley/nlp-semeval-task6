from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import TweetTokenizer
from sklearn.model_selection import cross_val_score
from sklearn import svm
from nltk.data import load
from nltk.tokenize import TweetTokenizer

import pandas
import nltk
import numpy as np

TENSED = True

TENSED_POS_TAGS = ['VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
STANCES = ['FAVOR', 'NONE', 'AGAINST']
# TAG_SET = {}
tagdict = load('help/tagsets/upenn_tagset.pickle')
TAG_SET = list(tagdict.keys())
TAG_SET.append('#')


def listOfTopics(dataframe):
    listOfTopics = []
    for row in traindf['Target']:
        if row not in listOfTopics:
            listOfTopics.append(row)
    return listOfTopics


def pos_tag(s):
    s = ' '.join([word for word in TweetTokenizer().tokenize(s) if word and '#' not in word])
    return nltk.pos_tag(nltk.word_tokenize(s))


def add_to_tag_set(s):
    tags = nltk.pos_tag(nltk.word_tokenize(s['Tweet']))
    for tag in tags:
        if tag[1] in TAG_SET:
            TAG_SET[tag[1]] = TAG_SET[tag[1]] + 1
        else:
            TAG_SET[tag[1]] = 1


def get_tag_set(rows):
    list(map(add_to_tag_set, rows))


def tag_is_tensed(tag):
    return tag[1] in TENSED_POS_TAGS


def get_target_rows_in_frame(d_frame, target):
    if target == 'ALL':
        return list(
            map((lambda x: x[1]), d_frame.iterrows()))
    return list(
        filter((lambda row: row['Target'] == target),
               map((lambda x: x[1]), d_frame.iterrows())))


def get_target_rows(filename, target):
    training_df = pandas.read_csv(filename, sep='\t', encoding='latin1')
    return get_target_rows_in_frame(training_df, target)


def get_tagged_words(rows):
    return list(map((lambda row: pos_tag(row['Tweet'])), rows))


def tag_counts(tags):
    just_tags = list(map((lambda x: x[1]), tags))
    return list(map((lambda tag: just_tags.count(tag)), TAG_SET))


def tensed_tag_counts(tags):
    just_tags = list(map((lambda x: x[1]), tags))
    return list(map((lambda tag: just_tags.count(tag)), TENSED_POS_TAGS))


def get_x_train(tags_for_tweets, tag_count_func):
    return list(map(tag_count_func, tags_for_tweets))


def onehot_for_stance(stance):
    one_hot = [0, 0, 0]
    one_hot[STANCES.index(stance)] = 1
    return one_hot


def get_y_train(rows):
    return list(
        map(onehot_for_stance,
            map((lambda row: row['Stance']), rows)))


def get_xy_data(filename, target, tensed=False):
    target_rows = get_target_rows(filename, target)
    # Get tags
    #     get_tag_set(target_rows)

    tag_count_func = tag_counts
    if tensed:
        tag_count_func = tensed_tag_counts

    tagged_tweets = get_tagged_words(target_rows)
    x_train = get_x_train(tagged_tweets, tag_count_func)
    y_train = get_y_train(target_rows)
    return (x_train, y_train)


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

outdf = pandas.DataFrame(columns=['ID', 'Target', 'Tweet', 'Stance'])

for topic in listOfTopics(traindf):
    extract = traindf.loc[traindf['Target'] == topic]
    corpus = list(map(lambda x: x[:-6], list(extract['Tweet'])))
    Y = list(extract['Stance'])
    ngram_vectorizer = CountVectorizer(ngram_range=(1, 1), analyzer='word', min_df=3, tokenizer=lambda x: custToken(x))
    (x_train, _) = get_xy_data('trainingdata.txt', topic, TENSED)
    X = ngram_vectorizer.fit_transform(corpus).toarray()
    X = np.concatenate((X, np.array(x_train)), axis=1)
    clf = svm.SVC(kernel='linear')
    clf.fit(X, Y)

    testExtract = testdf.loc[testdf['Target'] == topic]
    testCorpus = list(map(lambda x: x[:-6], list(testExtract['Tweet'])))
    (x_test, _) = get_xy_data('subtaskA-testdata-gold.txt', topic, TENSED)
    testX = ngram_vectorizer.transform(testCorpus).toarray()
    testX = np.concatenate((testX, np.array(x_test)), axis=1)

    predictions = clf.predict(testX)

    ID = list(map(lambda x: str(x), list(testExtract['ID'])))
    s = zip(ID, list(testExtract['Target']), testCorpus, list(predictions))

    for x in s:
        outdf.loc[len(outdf)] = list(x)


outdf.set_index('ID')
outdf.to_csv('output.txt', sep='\t', index=False)
