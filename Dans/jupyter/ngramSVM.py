
# coding: utf-8

# In[1]:



# First we need to import data into pandas dataframe

# In[2]:

import pandas
df = pandas.read_csv('trainingData.txt', sep='\t', encoding='latin1', index_col='ID')
df.head()


# Then extract the topics (for now just focus on hillary)
# 

# In[3]:

listOfTopics = []
for row in df['Target']:
    if row not in listOfTopics:
        listOfTopics.append(row)
listOfTopics

topic = 'Legalization of Abortion'

# Create a function to turn the stance into usable values

# In[4]:

def yVec(text):
    if text == 'AGAINST':
        return -1
    elif text == 'FAVOR':
        return 1
    else:
        return 0


# Extract topic rows

# In[5]:

trainDF = df.loc[df['Target'] == topic]
trainDF.describe()


# Make corpus from tweets

# In[6]:

corpus = []
Y = []
for index, row in trainDF.iterrows():
    corpus.append(row['Tweet'][:-6])
    Y.append(yVec(row['Stance']))


# Generate ngrams of size 2-4.  Drop anything with less then 3 occurences 

# In[7]:

from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import TweetTokenizer

ngram_vectorizer = CountVectorizer(ngram_range=(1, 7), analyzer='word', stop_words='english', min_df=2, tokenizer=lambda x: TweetTokenizer().tokenize(x))

X = ngram_vectorizer.fit_transform(corpus).toarray()


# So what does a single tweet now look like?  Following the progression we've done

# In[8]:

print('Tweet: \n', corpus[0])
print(TweetTokenizer().tokenize(corpus[0]))
print('---------------------------------------------------------------------------------\n')
print('Total features from corpus: ', len(ngram_vectorizer.get_feature_names()))
print('First 20: \n', ngram_vectorizer.get_feature_names()[:20])
print('---------------------------------------------------------------------------------\n')
print('Our tweet represented as a vector: \n')
print(X[0])
print('---------------------------------------------------------------------------------\n')
print('\nOur X: \n')
print('Shape: ', X.shape)


# Time to apply a svm

# In[12]:

testFullDF = pandas.read_csv('test.txt', sep='\t', encoding='latin1', index_col='ID')
testFullDF.head()

testDF = df.loc[df['Target'] == topic]


# In[14]:

test_corpus = []
test_Y = []
for index, row in testDF.iterrows():
    test_corpus.append(row['Tweet'][:-6])
    test_Y.append(yVec(row['Stance']))

test_X = ngram_vectorizer.transform(test_corpus).toarray()


# In[15]:

from sklearn.model_selection import cross_val_score
from sklearn import svm
clf = svm.SVC(kernel='poly', C=1, random_state=1)
scores = cross_val_score(clf, X, Y, cv=15)


# And our score on quick test set:

# In[16]:

print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# Trying adaboost

# In[17]:

from sklearn.ensemble import AdaBoostClassifier

test = AdaBoostClassifier(svm.SVC(probability=True, kernel='poly'), n_estimators=50, learning_rate=1.0, algorithm='SAMME')
scores = cross_val_score(test, X, Y, cv=5)
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# In[ ]:



