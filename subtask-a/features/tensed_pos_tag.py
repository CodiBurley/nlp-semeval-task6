from nltk.data import load

import nltk
import pandas

TENSED_POS_TAGS = ['VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
STANCES = ['FAVOR', 'NONE', 'AGAINST']
# TAG_SET = {}
tagdict = load('help/tagsets/upenn_tagset.pickle')
TAG_SET = list(tagdict.keys())
TAG_SET.append('#')


def pos_tag(s):
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