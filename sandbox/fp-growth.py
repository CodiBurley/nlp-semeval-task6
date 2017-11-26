""" This runs fp-growth on all 5 stances
    listed within the testing set given to us"""

from lib.FPTree import buildFPTree
import pandas


def ListOfTopics(dataframe):
    lst = []
    for row in dataframe['Target']:
        if row not in lst:
            lst.append(row)
    return lst


def treeparse(tree):
    ret = []
    for x in tree:
        if x.freq > 3:
            ret.append(tuple(x.items))
    return ret


def treediff(a,f,n):
    a = set(treeparse(a))
    f = set(treeparse(f))
    n = set(treeparse(n))
    
    return (a | f | n) - (a & f & n)


class Holder:
    def __init__(self, topic, against, favor, none):
        self.topic = topic
        self.against = against
        self.favor = favor
        self.none = none


if __name__ == "__main__":

    traindf = pandas.read_csv('Dans/trainingData.txt',
                              sep='\t', encoding='latin1')
    print(traindf.head())

    trees = []
    for topic in ListOfTopics(traindf):
        print(topic)
        extract = traindf.loc[traindf['Target'] == topic]
        extract['Tweet'] = extract['Tweet'].apply(lambda x: x[:-6])

        against = extract.loc[extract['Stance'] == 'AGAINST']
        against = against['Tweet']

        favor = extract.loc[extract['Stance'] == 'Favor']
        favor = favor['Tweet']

        none = extract.loc[extract['Stance'] == 'None']
        none = none['Tweet']

        againstTree = buildFPTree(against)
        favorTree = buildFPTree(favor)
        noneTree = buildFPTree(none)

        tree = treediff(againstTree,favorTree,noneTree)

        print(tree)
