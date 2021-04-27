import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from load_data import load_twitter_data, TwitterDataset
from cnn_model import CNN
from plot_results import plot_accuracy, plot_losses
import tqdm
import argparse


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


def eval_network(data, net, use_gpu=False, batch_size=25, device=torch.device('cpu')):
    print("Evaluation Set")
    num_correct = 0
    Y = (data.labels + 1.0) / 2.0
    X = data.Xwordlist
    batch_predictions = []
    for batch in tqdm.tqdm(range(0, len(X), batch_size), leave=False):
        batch_x = pad_batch_input(X[batch:batch + batch_size], device=device)
        batch_y = torch.tensor(Y[batch:batch + batch_size], device=device)
        # if use_gpu and torch.cuda.is_available():
        #     batch_x = batch_x.cuda()
        batch_y_hat = net.forward(batch_x)
        predictions = batch_y_hat.argmax(dim=1)
        batch_predictions.append(predictions)
        # num_correct = float((predictions == batch_y).sum())
        # accuracy = num_correct/float(batch_size)
        # batch_accuracies.append(accuracy)
    predictions = torch.cat(batch_predictions)
    Y_tensor = torch.tensor(Y, device=device)
    num_correct = float((predictions == Y_tensor).sum())
    accuracy = num_correct/len(Y)

    print("Eval Accuracy: %s" % accuracy)
    return accuracy
    # return min_accuracy, accuracy, max_accuracy

def convert_to_onehot(Y_list, NUM_CLASSES=2, device=torch.device('cpu')):
    Y_onehot = torch.zeros((len(Y_list), NUM_CLASSES), device=device)
    # Y_onehot = [torch.zeros(len(l), NUM_CLASSES) for l in Y_list]
    for i in range(len(Y_list)):
        Y_onehot[i, int(Y_list[i])]= 1.0
    return Y_onehot


def pad_batch_input(X_list, device=torch.device('cpu')):
    X_padded = torch.nn.utils.rnn.pad_sequence([torch.as_tensor(l) for l in X_list], batch_first=True).type(torch.LongTensor).to(device)
    # X_mask   = torch.nn.utils.rnn.pad_sequence([torch.as_tensor([1.0] * len(l)) for l in X_list], batch_first=True).type(torch.FloatTensor)
    return X_padded



def train_network(net, X, Y, num_epochs, dev, lr=0.001, batchSize=50, use_gpu=False, device=torch.device('cpu')):

    print("Start Training!")
    #TODO: initialize optimizer.
    optimizer = optim.Adam(net.parameters(), lr=lr)
    num_classes = len(set(Y))
    epoch_losses = []
    eval_accuracy = []
    # min_eval_accuracies = []
    # max_eval_accuracies = []
    for epoch in range(num_epochs):
        num_correct = 0
        total_loss = 0.0
        net.train()   #Put the network into training model
        for batch in tqdm.tqdm(range(0, len(X), batchSize), leave=False):
            batch_tweets = X[batch:batch + batchSize]
            batch_labels = Y[batch:batch + batchSize]
            batch_tweets = pad_batch_input(batch_tweets, device=device)
            batch_onehot_labels = convert_to_onehot(batch_labels, NUM_CLASSES=num_classes, device=device)
            optimizer.zero_grad()
            batch_y_hat = net.forward(batch_tweets)
            batch_losses = torch.neg(batch_y_hat)*batch_onehot_labels #cross entropy loss
            loss = batch_losses.mean()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().item())

        epoch_losses.append(total_loss)
        net.eval()    #Switch to eval mode
        print(f"loss on epoch {epoch} = {total_loss}")
        # min_acc, accuracy, max_acc = eval_network(dev, net, use_gpu=use_gpu, batch_size=batchSize, device=device)
        accuracy = eval_network(dev, net, use_gpu=use_gpu, batch_size=batchSize, device=device)
        eval_accuracy.append(accuracy)
        # min_eval_accuracies.append(min_acc)
        # max_eval_accuracies.append(max_acc)


    print("Finished Training")
    return epoch_losses, eval_accuracy


def main():
    args = parse_args()
    twitter_csv_path = args.tweet_csv_file
    device_type = args.device
    use_bert = False
    shuffle = False
    train_data, dev_data, test_data = load_twitter_data(twitter_csv_path, test_split_percent=0.1, val_split_percent=0.2, overfit=True, shuffle=shuffle, use_bert=use_bert, overfit_val=12639)
    vocab_size = train_data.vocab_size
    print(vocab_size)
    print(train_data.length)
    print(dev_data.length)
    print(test_data.length)
    cnn_net = CNN(vocab_size, DIM_EMB=300, NUM_CLASSES = 2)
    if device_type == "gpu" and torch.cuda.is_available():
        device = torch.device('cuda:0')
        cnn_net = cnn_net.cuda()
        epoch_losses, eval_accuracy = train_network(cnn_net,
                                        train_data.Xwordlist,
                                        (train_data.labels + 1.0)/2.0,
                                        10, dev_data, lr=0.003,
                                        batchSize=150, use_gpu=True, device=device)
        cnn_net.eval()
        print("Test Set")
        test_accuracy = eval_network(test_data, cnn_net, use_gpu=True, device=device)

    else:
        device = torch.device('cpu')
        epoch_losses, eval_accuracy = train_network(cnn_net,
                                        train_data.Xwordlist,
                                        (train_data.labels + 1.0)/2.0,
                                        10, dev_data, lr=0.003,
                                        batchSize=150, use_gpu=False, device=device)
        cnn_net.eval()
        print("Test Set")
        test_accuracy = eval_network(test_data, cnn_net, use_gpu=False, batch_size=batchSize, device=device)

    # plot_accuracy((min_accs, eval_accuracy, max_accs), "Sentiment CNN lr=0.001", train_data.length)
    plot_accuracy(eval_accuracy, "Sentiment CNN lr=0.003", train_data.length)
    plot_losses(epoch_losses, "Sentiment CNN lr=0.003", train_data.length)
    torch.save(cnn_net.state_dict(), "saved_models\\cnn.pth")
    np.save("cnn_train_loss_" + str(train_data.length) +  ".npy", np.array(epoch_losses))
    np.save("cnn_validation_accuracy_" + str(train_data.length) +  ".npy", np.array(eval_accuracy))
    # np.save("cnn_validation_accuracies.npy", np.array([min_accs, max_accs, eval_accuracy]))




if __name__ == '__main__':
    main()
