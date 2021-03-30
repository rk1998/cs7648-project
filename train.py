import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from load_data import load_twitter_data, TwitterDataset
from cnn_model import CNN
import tqdm
import argparse
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tweet_csv_file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "gpu"],
        default="cpu",
    )
    args = parser.parse_args()
    return args


def eval_network(data, net, use_gpu=False):
    num_correct = 0
    Y = (data.labels + 1.0) / 2.0
    X = data.Xwordlist
    padded_X, mask = pad_batch_input(X)
    if use_gpu and torch.cuda.is_available():
        padded_X = padded_X.cuda()
    y_hat = net.forward(padded_X)
    predictions = y_hat.argmax(dim=1)
    diff = torch.abs(predictions.cpu() - Y)
    num_correct = torch.sum((diff == 0).int())
    # for i in range(0, len(X), batch_size):
    #
    #
    #     words_i = X[i]
    #     words_i = words_i.reshape((1, words_i.shape[0]))
    #     if torch.cuda.is_available():
    #       words = words_i.cuda()
    #     logProbs = net.forward(words)
    #     pred = torch.argmax(logProbs)
    #     if pred == Y[i]:
    #         num_correct += 1
    print("Eval Accuracy: %s" % (float(num_correct) / float(len(X))))
    return (float(num_correct) / float(len(X)))

def convert_to_onehot(Y_list, NUM_CLASSES=2):
    Y_onehot = torch.zeros((len(Y_list), NUM_CLASSES))
    # Y_onehot = [torch.zeros(len(l), NUM_CLASSES) for l in Y_list]
    for i in range(len(Y_list)):
        Y_onehot[i, int(Y_list[i])]= 1.0
    return Y_onehot


def pad_batch_input(X_list):
    X_padded = torch.nn.utils.rnn.pad_sequence([torch.as_tensor(l) for l in X_list], batch_first=True).type(torch.LongTensor)
    X_mask   = torch.nn.utils.rnn.pad_sequence([torch.as_tensor([1.0] * len(l)) for l in X_list], batch_first=True).type(torch.FloatTensor)
    return (X_padded, X_mask)



def train_network(net, X, Y, num_epochs, dev, batchSize=50, use_gpu=False):

    print("Start Training!")
    #TODO: initialize optimizer.
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    num_classes = len(set(Y))
    epoch_losses = []
    eval_accuracy = []
    for epoch in range(num_epochs):
        num_correct = 0
        total_loss = 0.0
        net.train()   #Put the network into training model
        for batch in tqdm.tqdm(range(0, len(X), batchSize), leave=False):
            batch_tweets = X[batch:batch + batchSize]
            batch_labels = Y[batch:batch + batchSize]
            batch_tweets, batch_mask = pad_batch_input(batch_tweets)
            batch_onehot_labels = convert_to_onehot(batch_labels, NUM_CLASSES=num_classes)
            if use_gpu and torch.cuda.is_available():
              batch_tweets = batch_tweets.cuda()
              batch_onehot_labels = batch_onehot_labels.cuda()
            net.zero_grad()
            batch_y_hat = net.forward(batch_tweets)
            # batch_y_hat = batch_y_hat*batch_mask
            batch_losses = torch.neg(batch_y_hat)*batch_onehot_labels
            loss = batch_losses.mean()
            # loss = torch.neg(batch_y_hat).dot(batch_onehot_labels) #cross entropy loss
            total_loss += loss
            loss.backward()
            optimizer.step()
            #TODO: compute gradients, do parameter update, compute loss.
        epoch_losses.append(total_loss)
        net.eval()    #Switch to eval mode
        print(f"loss on epoch {epoch} = {total_loss}")
        accuracy = eval_network(dev, net, use_gpu=use_gpu)
        eval_accuracy.append(accuracy)

    print("Finished Training")
    return epoch_losses, eval_accuracy


def main():
    args = parse_args()
    twitter_csv_path = args.tweet_csv_file
    device_type = args.device
    train_data, dev_data, test_data = load_twitter_data(twitter_csv_path, split_percent=0.002, overfit=False)
    print(train_data.length)
    print(dev_data.length)
    print(test_data.length)
    cnn_net = CNN(train_data.vocab.GetVocabSize(), DIM_EMB=300, NUM_CLASSES = 2)
    if device_type == "gpu" and torch.cuda.is_available():
        cnn_net = cnn_net.cuda()
        epoch_losses, eval_accuracy = train_network(cnn_net,
                                        train_data.Xwordlist,
                                        (train_data.labels + 1.0)/2.0,
                                        10, dev_data,
                                        batchSize=50, use_gpu=True)
    else:
        epoch_losses, eval_accuracy = train_network(cnn_net,
                                        train_data.Xwordlist,
                                        (train_data.labels + 1.0)/2.0,
                                        10, dev_data,
                                        batchSize=50, use_gpu=False)
if __name__ == '__main__':
    main()
