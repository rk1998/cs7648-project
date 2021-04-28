import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import nltk
from nltk import word_tokenize
from nltk.tokenize import TreebankWordTokenizer
nltk.download('punkt')
from tokenizers import BertWordPieceTokenizer

from collections import Counter
import os
import sys
from sklearn.model_selection import train_test_split

tokenizer = BertWordPieceTokenizer("bert-base-uncased-vocab.txt", lowercase=True)

def load_tweet_csv(tweet_csv_path, overfit=True, shuffle_data=True, overfit_val=5000):
    if overfit:
        data = pd.read_csv(tweet_csv_path, nrows=overfit_val)
    else:
        data = pd.read_csv(tweet_csv_path)
    labels = data['label'].values
    labels[labels == 0] = -1
    labels[labels == 2] = 0
    labels[labels == 4] = 1
    tweets = data['text'].values
    if shuffle_data:
        indices = np.arange(tweets.shape[0])
        np.random.shuffle(indices)
        tweets = tweets[indices]
        labels = labels[indices]
    # if overfit:
    #     tweets = tweets[0:overfit_val]
    #     labels = labels[0:overfit_val]
    #     return labels, tweets
    # tweet_lists = split_tweets_to_lists(tweets.values)
    return labels, tweets

def load_unlabeled_tweet_csv(tweet_csv_path, all_tweets=False, num_tweets=50000):
    if all_tweets:
        data = pd.read_csv(tweet_csv_path)
    else:
        data = pd.read_csv(tweet_csv_path, nrows=num_tweets)
    tweets = data['text'].values
    labels = data['label'].values
    labels[labels == 0] = -1
    labels[labels == 2] = 0
    labels[labels == 4] = 1
    return tweets, labels

def split_data(tweet_csv_path, test_split_percent=0.2, val_split_percent=0.2, shuffle=True, overfit=False, overfit_val=5000):
    '''
    Splits Twitter Data into Training, Dev, and Test sets
    returns them as pandas dataframes
    '''
    labels, tweets = load_tweet_csv(tweet_csv_path, overfit=overfit, shuffle_data=shuffle, overfit_val=overfit_val)
    vocab = create_vocab(tweets)
    # indices = np.arange(tweets.shape[0])
    # np.random.shuffle(indices)
    # labels = labels[indices]
    # tweets = tweets[indices]
    X_train, X_test, y_train, y_test = train_test_split(tweets, labels, test_size=test_split_percent)
    test_data = pd.DataFrame({'label': y_test, 'text':X_test})
    X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=val_split_percent)
    dev_data = pd.DataFrame({'label': y_dev, 'text':X_dev})
    train_data = pd.DataFrame({'label':y_train, 'text':X_train})
    return train_data, dev_data, test_data, vocab


def create_vocab(tweet_data):
    vocab = Vocab()
    for tweet in tweet_data:
        tokenized_tweet = word_tokenize(tweet)
        for word in tokenized_tweet:
            id = vocab.GetID(word.lower())
    vocab.Lock()
    return vocab




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


    def convert_to_words(self, word_ids):
        """
        Converts a list of word ids to their actual words in the vocabulary
        Inputs:
        word_ids: list(int) - list of word ids
        Returns:
        str: the output string from the list of word ids
        """
        output = ""
        for i in range(len(word_ids)):
            word_i = self.GetWord(word_ids[i])
            if i == 0:
                output = word_i
            else:
                output = output + " " + word_i
        return output

    def Lock(self):
        self.locked = True

class TwitterDataset:
    '''
    Class to that tokenizes raw tweet text and stores corresponding labels
    '''
    def __init__(self, data_frame, vocab = None, use_bert_tokenizer=False):

        # labels, tweet_list = load_tweet_csv(twitter_csv_path)
        self.labels = data_frame['label'].values
        tweet_list = data_frame['text'].values
        self.length = len(self.labels)
        self.use_bert_tokenizer = use_bert_tokenizer
        # self.tweet_list = tweet_list
        if not vocab:
            self.vocab = Vocab()
        else:
            self.vocab = vocab

        self.Xwordlist = []
        if self.use_bert_tokenizer:
            for tweet in tweet_list:
                wordlist = tokenizer.encode(tweet).ids
                self.Xwordlist.append(wordlist)
        else:
            for tweet in tweet_list:
                wordlist = [self.vocab.GetID(w.lower()) for w in word_tokenize(tweet) if self.vocab.GetID(w.lower()) >= 0]
                self.Xwordlist.append(wordlist)

        if self.use_bert_tokenizer:
            self.vocab_size = tokenizer.get_vocab_size()
        else:
            self.vocab_size = self.vocab.GetVocabSize()

        self.vocab.Lock()
        index = np.arange(len(self.Xwordlist))
        np.random.shuffle(index) #randomly shuffle words and labels
        self.Xwordlist = [torch.LongTensor(self.Xwordlist[i]) for i in index]
        self.labels = self.labels[index]

    def convert_text_to_ids(self, text_list):
        id_list = []
        if self.use_bert_tokenizer:
            for item in text_list:
                wordlist = tokenizer.encode(item).ids
                # wordlist = [self.vocab.GetID(w.lower()) for w in word_tokenize(item) if self.vocab.GetID(w.lower()) >= 0]
                id_list.append(wordlist)
        else:
            for item in text_list:
                # wordlist = tokenizer.encode(item).ids
                word_tokens = word_tokenize(item)
                wordlist = []
                for w in word_tokens:
                    id = self.vocab.GetID(w.lower())
                    if id >= 0:
                        wordlist.append(id)
                # wordlist = [self.vocab.GetID(w.lower()) for w in word_tokenize(item) if self.vocab.GetID(w.lower()) >= 0]
                id_list.append(wordlist)
        id_list = [torch.LongTensor(id_list[i]) for i in range(0, len(id_list))]
        return id_list

    def convert_to_words(self, id_list):
        if self.use_bert_tokenizer:
            tweet = tokenizer.decode(id_list)
        else:
            output = ""
            for i in range(len(id_list)):
                word_i = self.vocab.GetWord(id_list[i])
                if i == 0:
                    output = word_i
                else:
                    output = output + " " + word_i
            return output
        return tweet


def load_twitter_data(tweet_filepath, test_split_percent=0.2, val_split_percent=0.2, shuffle=True, overfit=False, use_bert=False, overfit_val=500):
    '''
    Loads twitter csv file, splits it into training, dev, and test data
    and returns them as TwitterDataset objects.

    '''
    print("Splitting Data")
    train_data, dev_data, test_data, vocab = split_data(tweet_filepath, test_split_percent=test_split_percent, shuffle=shuffle, val_split_percent=val_split_percent, overfit=overfit, overfit_val=overfit_val)
    print("Converting to Indices")
    if not use_bert:
        train_dataset = TwitterDataset(train_data, vocab=vocab)
        dev_dataset = TwitterDataset(dev_data, vocab=vocab)
        test_dataset = TwitterDataset(test_data, vocab=vocab)
    else:
        train_dataset = TwitterDataset(train_data, use_bert_tokenizer=use_bert)
        dev_dataset = TwitterDataset(dev_data, use_bert_tokenizer=use_bert)
        test_dataset = TwitterDataset(test_data, use_bert_tokenizer=use_bert)
    return train_dataset, dev_dataset, test_dataset

def load_twitter_data_active_learning(tweet_filepath, test_split_percent=0.2, val_split_percent=0.2, seed_size=1000, overfit=False, overfit_val=500):
    train_data, dev_data, test_data = split_data(tweet_filepath,
                                                 test_split_percent=test_split_percent,
                                                 val_split_percent=val_split_percent,
                                                 overfit=overfit,
                                                 overfit_val=overfit_val)

    train_dataset = TwitterDataset(train_data)
    seed_data = pd.DataFrame({'label':train_data['label'][0:seed_size], 'text':train_data['text'][0:seed_size]})
    unlabeled_data = pd.DataFrame({'label':train_data['label'][seed_size:], 'text':train_data['text'][seed_size:]})

    seed_dataset = TwitterDataset(seed_data, vocab=train_dataset.vocab)
    unlabeled_data = TwitterDataset(unlabeled_data, vocab=train_dataset.vocab)
    dev_dataset = TwitterDataset(dev_data, vocab=train_dataset.vocab)
    test_dataset = TwitterDataset(test_data, vocab=train_dataset.vocab)
    return seed_dataset, unlabeled_data, dev_dataset, test_dataset




def main():
    twitter_csv_path = "..\\twitter_test.csv"
    # train_dataset, dev_data, test_dataset = load_twitter_data(twitter_csv_path, split_percent=0.3, overfit=True)
    seed_dataset, unlabeled_dataset, dev_dataset, test_dataset = load_twitter_data_active_learning(twitter_csv_path, test_split_percent=0.2, overfit=True, overfit_val=12000)
    # tweet_data = TwitterDataset(twitter_csv_path)
    print(seed_dataset.length)
    print(unlabeled_dataset.length)
    print(dev_dataset.length)
    print(test_dataset.length)
    print(seed_dataset.Xwordlist[2].tolist())
    # print(train_dataset.Xwordlist[0].tolist())
    print([seed_dataset.vocab.GetWord(x) for x in seed_dataset.Xwordlist[2].tolist()])
    print(seed_dataset.labels[2])
    print(seed_dataset.labels[0:10])

if __name__ == '__main__':
    main()
