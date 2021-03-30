import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import nltk
from nltk import word_tokenize
nltk.download('punkt')

from collections import Counter
import os
import sys
from sklearn.model_selection import train_test_split


def load_tweet_csv(tweet_csv_path):
    data = pd.read_csv(tweet_csv_path)
    labels = data['label'].values
    labels[labels == 0] = -1
    labels[labels == 2] = 0
    labels[labels == 4] = 1
    tweets = data['text'].values
    # tweet_lists = split_tweets_to_lists(tweets.values)
    return labels, tweets

def split_data(tweet_csv_path, split_percent=0.2):
    '''
    Splits Twitter Data into Training, Dev, and Test sets
    returns them as pandas dataframes
    '''
    labels, tweets = load_tweet_csv(tweet_csv_path)
    indices = np.arange(tweets.shape[0])
    np.random.shuffle(indices)
    labels = labels[indices]
    tweets = tweets[indices]
    X_train, X_test, y_train, y_test = train_test_split(tweets, labels, test_size=split_percent)
    test_data = pd.DataFrame({'label': y_test, 'text':X_test})
    X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=split_percent)
    dev_data = pd.DataFrame({'label': y_dev, 'text':X_dev})
    train_data = pd.DataFrame({'label':y_train, 'text':X_train})
    return train_data, dev_data, test_data

class Vocab:
    '''
    Class that maps words in the twitter dataset to indices
    '''
    def __init__(self, vocabFile=None):
        self.locked = False
        self.nextId = 0
        self.word2id = {}
        self.id2word = {}
        if vocabFile:
            for line in open(vocabFile):
                line = line.rstrip('\n')
                (word, wid) = line.split('\t')
                self.word2id[word] = int(wid)
                self.id2word[wid] = word
                self.nextId = max(self.nextId, int(wid) + 1)

    def GetID(self, word):
        if not word in self.word2id:
            if self.locked:
                return -1        #UNK token is -1.
            else:
                self.word2id[word] = self.nextId
                self.id2word[self.word2id[word]] = word
                self.nextId += 1
        return self.word2id[word]

    def HasWord(self, word):
        return self.word2id.has_key(word)

    def HasId(self, wid):
        return self.id2word.has_key(wid)

    def GetWord(self, wid):
        return self.id2word[wid]

    def SaveVocab(self, vocabFile):
        fOut = open(vocabFile, 'w')
        for word in self.word2id.keys():
            fOut.write("%s\t%s\n" % (word, self.word2id[word]))

    def GetVocabSize(self):
        #return self.nextId-1
        return self.nextId

    def GetWords(self):
        return self.word2id.keys()

    def Lock(self):
        self.locked = True

class TwitterDataset:
    '''
    Class to that tokenizes raw tweet text and stores corresponding labels
    '''
    def __init__(self, data_frame, vocab = None):

        # labels, tweet_list = load_tweet_csv(twitter_csv_path)
        self.labels = data_frame['label'].values
        tweet_list = data_frame['text'].values
        self.length = len(self.labels)
        # self.tweet_list = tweet_list
        if not vocab:
            self.vocab = Vocab()
        else:
            self.vocab = vocab

        self.Xwordlist = []
        for tweet in tweet_list:
            wordlist = [self.vocab.GetID(w.lower()) for w in word_tokenize(tweet) if self.vocab.GetID(w.lower()) >= 0]
            self.Xwordlist.append(wordlist)

        self.vocab.Lock()
        index = np.arange(len(self.Xwordlist))
        np.random.shuffle(index) #randomly shuffle words and labels
        self.Xwordlist = [torch.LongTensor(self.Xwordlist[i]) for i in index]
        self.labels = self.labels[index]


def load_twitter_data(tweet_filepath, split_percent=0.2, overfit=False):
    '''
    Loads twitter csv file, splits it into training, dev, and test data
    and returns them as TwitterDataset objects.

    '''
    print("Splitting Data")
    train_data, dev_data, test_data = split_data(tweet_filepath, split_percent=split_percent)

    print("Converting to Indices")
    if overfit:
        print("Returning Overfit set")
        train_dataset = TwitterDataset(train_data[0:15000])
        dev_dataset = TwitterDataset(dev_data[0:15000], vocab=train_dataset.vocab)
        test_dataset = TwitterDataset(test_data[0:15000], vocab=train_dataset.vocab)
    else:
        train_dataset = TwitterDataset(train_data)
        dev_dataset = TwitterDataset(dev_data, vocab=train_dataset.vocab)
        test_dataset = TwitterDataset(test_data, vocab=train_dataset.vocab)
    return train_dataset, dev_dataset, test_dataset



def main():
    twitter_csv_path = "..\\twitter_test.csv"
    train_dataset, dev_data, test_dataset = load_twitter_data(twitter_csv_path, split_percent=0.3, overfit=True)
    # tweet_data = TwitterDataset(twitter_csv_path)
    print(train_dataset.length)
    print(dev_data.length)
    print(test_dataset.length)
    print(train_dataset.Xwordlist[0].tolist())
    print([train_dataset.vocab.GetWord(x) for x in train_dataset.Xwordlist[0].tolist()])
    print(train_dataset.labels[0])

if __name__ == '__main__':
    main()
