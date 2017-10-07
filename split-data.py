import sys
import pandas
import nltk
import keras

from keras.models import Sequential
from keras.layers import Dense, Activation
from numpy import array as np_array

print(sys.argv[1])

TENSED_POS_TAGS = [ 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ' ]
STANCE_INDICES = { 'FAVOR': 0, 'NONE': 1, 'AGAINST': 2 }


def pos_tag(s):
    return nltk.pos_tag(nltk.word_tokenize(s))

def tag_is_tensed(tag):
    return tag[1] in TENSED_POS_TAGS


def get_target_rows(d_frame, target):
    return list(
        filter((lambda row: row['Target'] == target),
            map((lambda x: x[1]), d_frame.iterrows())))

def get_tensed_tagged_words(rows):
    return list(
        map(
            (lambda row:
                list(filter(tag_is_tensed, pos_tag(row['Tweet'])))),
            rows))


def tensed_tag_counts(tags):
    just_tags = list(map((lambda x: x[1]), tags))
    return list(map((lambda tag: just_tags.count(tag)), TENSED_POS_TAGS))

def get_x_train(tags_for_tweets):
    return list(map(tensed_tag_counts, tags_for_tweets))


def onehot_for_stance(stance):
    one_hot = [0,0,0]
    one_hot[STANCE_INDICES[stance]] = 1
    return one_hot

def get_y_train(rows):
    return list(
        map(onehot_for_stance,
            map((lambda row: row['Stance']), rows)))

def get_xy_data(filename):
    training_df = pandas.read_csv(filename, sep='\t', encoding='latin1')
    hillary_rows = get_target_rows(training_df, 'Hillary Clinton')
    tensed_tagged_tweets = get_tensed_tagged_words(hillary_rows)
    x_train = get_x_train(tensed_tagged_tweets)
    y_train = get_y_train(hillary_rows)
    return (x_train, y_train)

(x_train, y_train) = get_xy_data(sys.argv[1])
(x_test, y_test) = get_xy_data(sys.argv[2])


# LEARN
model = Sequential()
model.add(Dense(units=8, input_dim=len(TENSED_POS_TAGS))) # layer
model.add(Activation('relu')) # layer
model.add(Dense(units=16))
model.add(Activation('relu'))
model.add(Dense(units=3)) # layer
model.add(Activation('softmax')) #layer

model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.SGD(lr=0.03, momentum=0.9, nesterov=True),
              metrics=['accuracy', 'mae']) # not layer

model.fit(np_array(x_train), np_array(y_train), epochs=100, batch_size=32)

loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
print(loss_and_metrics)
