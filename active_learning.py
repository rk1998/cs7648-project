import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from load_data import load_unlabeled_tweet_csv, load_twitter_data, TwitterDataset, Vocab
from cnn_model import CNN
from plot_results import plot_accuracy, plot_losses
import tqdm
import argparse
from train import eval_network, pad_batch_input, convert_to_onehot
import pandas as pd



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labeled_tweet_csv_file",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--unlabeled_tweet_csv_file",
        type=str,
        required=True
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "gpu"],
        default="cpu",
    )
    parser.add_argument("--acquisition_func", type=str, choices=["least_confidence"], default="least_confidence")
    parser.add_argument("--seed_data_size", type=int, default=1000)
    args = parser.parse_args()
    return args





def label_data(data_samples, vocab):
    """
    List of input tweets selected by sampling strategy to be hand-labeled
    Inputs:
    data_sample: list(tuople) - list of tokenized tweets and their scores that
    vocab: Vocab - object containing the vocabulary mappings of the tweet data
    Returns:
    list(int) - list of sentiment labels for each tweet (either 0 or 1)
    """
    print("For each tweet, enter 0 if the tweet has a negative sentiment or enter 1 if it has a positive sentiment")
    labels = []
    for i in range(len(data_samples)):
        sample_i = data_samples[i][0]
        tweet = vocab.convert_to_words(sample_i)
        print(tweet + "\n")
        label = input("Enter Sentiment:  ")
        try:
            label = int(label)
        except:
            label=1
        if label != 1 and label != 0:
            label = 1
        labels.append(label)
    print(labels)
    print(len(labels))
    return labels


def least_confidence(model_outputs, num_classes=2):
    """
    Acquistion Function for Active Learning Pipeline:
    Computes the least confidence metric across batch of model outputs.
    Inputs:
    model_outputs: torch.tensor of size (batch_size x 2)
    Returns: torch.tensor, normalized, least confidence scores for the model's
    class probability distribution
    """
    max_scores, indices = torch.max(model_outputs, dim=1)
    confidence_scores = 1.0 - max_scores
    normalized_scores = confidence_scores*(num_classes/(num_classes - 1))
    return normalized_scores

def compute_acquisition_function(model, acquisition_function, X, num_samples=100, batch_size=50, reverse=False, device=torch.device('cpu')):
    """
    Computes the acquisition_function on the unlabeled data samples to
    determine which samples will be hand labeled.
    Inputs:

    model - torch.nn module: Neural Network model to pass inputs into and get probability scores from
    acquisition_function - function to compute acquistion scores on model outputs.
    This score will be used to rank the input tweets and determine which tweets will
    be hand labeled next
    X - torch.LongTensor Unlabeled model inputs
    num_samples - number of samples to select after scoring and ranking
    batch_size - size of batch for model computation
    reverse: bool - flag used to sort acquisition scores in descending order
    device: torch.device - pytorch device object
    Returns:
    list(tuple) - list of tuples (score, input tweet) ranked from lowest to highest score (or opposite if reverse = True)
    """
    scored_samples = []
    print("Scoring Unlabeled Data")
    model.eval()
    for batch in tqdm.tqdm(range(0, len(X), batch_size), leave=False):
        batch_sentences = X[batch:batch + batch_size]
        padded_batch = pad_batch_input(batch_sentences, device=device)
        batch_scores = model.forward(padded_batch)
        acquisition_scores = acquisition_function(batch_scores)
        for j in range(0, len(batch_sentences)):
            input_sentence = batch_sentences[j]
            input_sentence = input_sentence.tolist()
            score = acquisition_scores[j].item()
            scored_samples.append((input_sentence, score))

    scored_samples.sort(reverse=reverse, key=lambda x: x[1])
    if num_samples > len(scored_samples):
        return samples, []
    else:
        samples = scored_samples[0:num_samples]
        unused_samples = scored_samples[num_samples: ]
        X_unlabeled = [torch.LongTensor(sentence) for sentence, score in unused_samples]
        return samples, X_unlabeled


def train_step(net, X, Y, epoch_num, dev, optimizer, num_classes=2, batchSize=50, use_gpu=False, device=torch.device('cpu')):
    """
    Performs one supervised training epoch on batches of data
    """
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

    net.eval()    #Switch to eval mode
    print(f"loss on epoch {epoch_num} = {total_loss}")
    # min_acc, accuracy, max_acc = eval_network(dev, net, use_gpu=use_gpu, batch_size=batchSize, device=device)
    accuracy = eval_network(dev, net, use_gpu=use_gpu, batch_size=batchSize, device=device)
    return total_loss, accuracy




def train_active_learning(net, vocab, X_seed, Y_seed, X_unlabeled, dev, num_epochs=15, acquisition_func=least_confidence, lr=0.001, batchSize=50, num_samples=100, use_gpu=False, device=torch.device('cpu')):
    """
    Main active learning training loop
    """
    print("Start Active Learning Training")
    optimizer = optim.Adam(net.parameters(), lr=lr)
    num_classes = len(set(Y_seed))
    epoch_losses = []
    eval_accuracy = []
    hand_labeled_data = []
    for epoch in range(num_epochs):
        print("EPOCH: " + str(epoch))
        total_loss, accuracy = train_step(net, X_seed, Y_seed, epoch, dev, optimizer, num_classes=num_classes, batchSize=batchSize, use_gpu=use_gpu, device=device)
        epoch_losses.append(total_loss)
        eval_accuracy.append(accuracy)
        if len(X_unlabeled) > 0:
            samples_to_label, X_unlabeled = compute_acquisition_function(net, acquisition_func, X_unlabeled, num_samples=num_samples, batch_size=batchSize, device=device)
            new_labels = label_data(samples_to_label, vocab)
            X_samples = [torch.LongTensor(sample) for sample, score in samples_to_label]
            for i in range(len(samples_to_label)):
                sample = samples_to_label[i]
                label = new_labels[i]
                hand_labeled_data.append((sample, label))
            for sample_tensor in X_samples:
                X_seed.append(sample_tensor)
            Y_seed = np.concatenate((Y_seed, new_labels))
            print(Y_seed)
            index = np.arange(len(X_seed))
            np.random.shuffle(index) #randomly shuffle words and labels
            X_seed = [X_seed[i] for i in index]
            Y_seed = Y_seed[index]
        else:
            print("All Data Labelled")

    print("Finished Training")

    return epoch_losses, eval_accuracy, hand_labeled_data

def main():
    args = parse_args()
    # twitter_csv_path = args.tweet_csv_file
    labeled_twitter_csv_path = args.labeled_tweet_csv_file
    unlabeled_twitter_csv_path = args.unlabeled_tweet_csv_file

    device_type = args.device
    acquistion_function_type = args.acquisition_func

    if acquistion_function_type == "least_confidence":
        acquisition_func = least_confidence
    else:
        acquisition_func = least_confidence


    seed_data_size = args.seed_data_size
    train_data, dev_data, test_data = load_twitter_data(labeled_twitter_csv_path, test_split_percent=0.1, val_split_percent=0.2, overfit=True, overfit_val=50000)
    vocab = train_data.vocab
    print(vocab.GetVocabSize())
    unlabeled_tweets = load_unlabeled_tweet_csv(unlabeled_twitter_csv_path)
    X_unlabeled = train_data.convert_text_to_ids(unlabeled_tweets)[0:70000]

    X_seed = train_data.Xwordlist[0:seed_data_size]
    Y_seed = train_data.labels[0:seed_data_size]
    Y_seed = (Y_seed + 1.0)/2.0


    cnn_net = CNN(train_data.vocab.GetVocabSize(), DIM_EMB=300, NUM_CLASSES = 2)
    if device_type == "gpu" and torch.cuda.is_available():
        device = torch.device('cuda:0')
        cnn_net = cnn_net.cuda()
        epoch_losses, eval_accuracy, hand_labeled_data = train_active_learning(cnn_net, vocab,
                                                            X_seed, Y_seed,
                                                            X_unlabeled, dev_data,
                                                            num_epochs=10, acquisition_func=acquisition_func,
                                                            lr=0.003, batchSize=150, num_samples=15,
                                                            use_gpu=True, device=device)
        cnn_net.eval()
        print("Test Set")
        test_accuracy = eval_network(test_data, cnn_net, use_gpu=True, device=device)

    else:
        device = torch.device('cpu')
        cnn_net = cnn_net.cuda()
        epoch_losses, eval_accuracy, hand_labeled_data = train_active_learning(cnn_net, vocab,
                                                            X_seed, Y_seed,
                                                            X_unlabeled, dev_data,
                                                            num_epochs=10, acquisition_function=acquisition_func,
                                                            lr=0.003, batchSize=150, num_samples=15,
                                                            use_gpu=True, device=device)
        cnn_net.eval()
        print("Test Set")
        test_accuracy = eval_network(test_data, cnn_net, use_gpu=True, device=device)


    # plot_accuracy((min_accs, eval_accuracy, max_accs), "Sentiment CNN lr=0.001", train_data.length)
    plot_accuracy(eval_accuracy, "Sentiment CNN (Active Learning) lr=0.003 " + acquistion_function_type, train_data.length)
    plot_losses(epoch_losses, "Sentiment CNN (Active Learning) lr=0.003 " + acquistion_function_type, train_data.length)
    torch.save(cnn_net.state_dict(), "saved_models\\cnn.pth")
    np.save("cnn_active_learning_train_loss" + acquistion_function_type + "_" + str(seed_data_size) + ".npy", np.array(epoch_losses))
    np.save("cnn_active_learning_validation_accuracy" + acquistion_function_type + "_" + str(seed_data_size) + ".npy", np.array(eval_accuracy))

    human_labels = []
    tweets = []

    for sample, label in hand_labeled_data:
        tweet, score = sample
        tweet = vocab.convert_to_words(tweet)
        tweets.append(tweet)
        human_labels.append(label)

    new_labeled_tweets = pd.DataFrame({'label':human_labels, 'text':tweets})
    new_labeled_tweets.to_csv("human_labeled_tweets.csv", header=True, index=False)


    # np.save("cnn_validation_accuracies.npy", np.array([min_accs, max_accs, eval_accuracy]))




if __name__ == '__main__':
    main()
