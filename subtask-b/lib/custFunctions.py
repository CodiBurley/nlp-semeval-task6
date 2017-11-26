from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

import numpy as np
import nltk
import pandas


def HashtagToken(sent):
    tweet = TweetTokenizer().tokenize(sent)
    returnTweet = [w for w in tweet if '#' in w]
    return returnTweet

def tfidfVector(tweets):
    vectorizer = TfidfVectorizer(stop_words='english')
    return vectorizer.fit_transform(tweets)

def HashtagVector(tweets):
    ngram_vectorizer = CountVectorizer(ngram_range=(1, 1), analyzer='word', stop_words='english', min_df=10, tokenizer=lambda x: HashtagToken(x))
    return ngram_vectorizer.fit_transform(tweets)


def adjVector(tweets):
    stops = set(stopwords.words("english"))
    tweets = tweets.apply(lambda x: x.lower())
    tweets = tweets.apply(lambda x: ' '.join([word for word in TweetTokenizer().tokenize(x) if 'http' not in word and 'https' not in word]))
    tweets = tweets.apply(lambda x: ' '.join([word for word in TweetTokenizer().tokenize(x) if '/' not in word]))
    tweets = tweets.apply(lambda x: ' '.join([word for word in TweetTokenizer().tokenize(x) if '\\x' not in word]))
    tweets = tweets.apply(lambda x: ' '.join([word for word in TweetTokenizer().tokenize(x) if word not in stops and '@' not in word]))
    tweets = tweets.apply(lambda x: ' '.join([word for word in TweetTokenizer().tokenize(x) if word not in stops and '#' not in word]))
    pos = map(lambda s: nltk.pos_tag(TweetTokenizer().tokenize(s)), tweets)
    pos = list(pos)

    adj = []
    for t in pos:
        for w in t:
            if 'JJ' in w[1]:
                adj.append(w[0])
                
    adj = sorted(set(adj))

    fit = []
    for tweet in tweets:
        ohv = [0] * len(adj)
        for word in TweetTokenizer().tokenize(tweet):
            if word in adj:
                ohv[adj.index(word)] += 1
        fit.append(ohv)
    return np.array(fit)