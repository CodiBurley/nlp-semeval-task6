from pyspark.mllib.fpm import FPGrowth
from nltk.tokenize import TweetTokenizer
from string import punctuation
from nltk.corpus import stopwords
from string import punctuation
from pyspark import SparkContext

import pandas

sc = SparkContext()
sc.setLogLevel("ERROR")

def __strip_punctuation(s):
    return ''.join(c for c in s if c not in punctuation.replace('#',''))

def buildFPTree(dataframe, minSupport=0.01, numPartitions=10, parallelizeSplit=5):
    availableTweets = dataframe.apply(lambda x: __strip_punctuation(x))

    availableTweets = availableTweets.apply(lambda x: x.lower())

    stops = set(stopwords.words("english"))
    availableTweets = availableTweets.apply(lambda x: ' '.join([word for word in TweetTokenizer().tokenize(x) if word not in stops and '@' not in word]))

    availableTweets = availableTweets.drop_duplicates()
   
    setized_availableTweets = []
    for row in availableTweets:
        seen = set()
        for word in TweetTokenizer().tokenize(row):
            if word not in seen:
                seen.add(word)
        setized_availableTweets.append(list(seen))

    transactions = sc.parallelize(setized_availableTweets, parallelizeSplit)
    model = FPGrowth.train(transactions, minSupport=minSupport, numPartitions=numPartitions)
    result = model.freqItemsets().collect()
    return result
