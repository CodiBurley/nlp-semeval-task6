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

def atToken(sent):
    tweet = TweetTokenizer().tokenize(sent)
    returnTweet = [w for w in tweet if '@' in w]
    return returnTweet

def tfidfVector(tweets):
    vectorizer = TfidfVectorizer(stop_words='english')
    return vectorizer.fit_transform(tweets)

def HashtagVector(tweets):
    hashtag_vectorier = TfidfVectorizer(ngram_range=(1, 1), analyzer='word', lowercase=True,stop_words='english', min_df=10, tokenizer=lambda x: HashtagToken(x))
    return hashtag_vectorier.fit_transform(tweets)

def atVector(tweets):
    hashtag_vectorier = TfidfVectorizer(ngram_range=(1, 1), analyzer='word', lowercase=True,stop_words='english', min_df=10, tokenizer=lambda x: atToken(x))
    return hashtag_vectorier.fit_transform(tweets)

def posToken(sent):
    token = nltk.pos_tag(TweetTokenizer().tokenize(sent))
    adj = []
    for w in token:
        if 'JJ' in w[1]:
            adj.append(w[0])
    return(adj)
     
def adjVector(tweets):
    stops = set(stopwords.words("english"))
    tweets = tweets.apply(lambda x: x.lower())
    tweets = tweets.apply(lambda x: ' '.join([word for word in TweetTokenizer().tokenize(x) if 'http' not in word and 'https' not in word]))
    tweets = tweets.apply(lambda x: ' '.join([word for word in TweetTokenizer().tokenize(x) if '/' not in word]))
    tweets = tweets.apply(lambda x: ' '.join([word for word in TweetTokenizer().tokenize(x) if word not in stops and '@' not in word]))
    tweets = tweets.apply(lambda x: ' '.join([word for word in TweetTokenizer().tokenize(x) if word not in stops and '#' not in word]))
    
    pos_vectorier = TfidfVectorizer(ngram_range=(1, 1), analyzer='word', lowercase=True,stop_words='english', min_df=10, tokenizer=lambda x: posToken(x))
    return pos_vectorier.fit_transform(tweets)