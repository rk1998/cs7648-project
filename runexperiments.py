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
    args = parser.parse_args()
    return args


def label_data(data_samples, vocab, use_human_labels=False):
    """
    List of input tweets selected by sampling strategy to be hand-labeled
    Inputs:
    data_sample: list(tuple) - list of tokenized tweets, their scores, and ground truth labels
    vocab: TwitterDataset - object containing the vocabulary mappings of the tweet data
    use_human_labels - if true, a person will provide labels, if false, use the ground truth labels
    Returns:
    list(int) - list of sentiment labels for each tweet (either 0 or 1)
    """
    print("For each tweet, enter 0 if the tweet has a negative sentiment or enter 1 if it has a positive sentiment")
    labels = []
    if use_human_labels:
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
    else:
        for i in range(len(data_samples)):
            label_i = data_samples[i][2]
            labels.append(label_i)
    print(labels)
    return labels

def random_score(model_outputs):
    """
    Acquisition Function for Active Learning Pipeline
    Assigns random scores to model outputs
    Inputs:
    model_outputs torch.tensor of size (batch_size x 2)
    Returns: torch.tensor or random scores between 0 and 1
    """
    return torch.rand(model_outputs.shape[0])

def entropy_score(model_outputs, num_classes=2):
    """
    Computes the normalized entropy of the model predictions
    Inputs:
    model_outputs: torch.tensor of size (batch_size x 2) - the models log_probs
    Returns: torch.tensor of size (batch_size x 2), returns normalized entropy scores
    based upon the model's predicted probabilities for each class
    """
    probs = torch.exp(model_outputs)
    log_probs = probs * torch.log2(probs)
    raw_entropy = 0 - torch.sum(log_probs, dim=1)
    normalized_entropy = raw_entropy/np.log2(num_classes)
    return normalized_entropy

def least_confidence(model_outputs, num_classes=2):
    """
    Acquistion Function for Active Learning Pipeline:
    Computes the least confidence metric across batch of model outputs.
    Inputs:
    model_outputs: torch.tensor of size (batch_size x 2)
    Returns: torch.tensor, normalized, least confidence scores for the model's
    class probability distribution
    """
    probs = torch.exp(model_outputs)
    max_scores, indices = torch.max(probs, dim=1)
    confidence_scores = 1.0 - max_scores
    normalized_scores = confidence_scores*(num_classes/(num_classes - 1))
    return normalized_scores

def compute_acquisition_function(model, acquisition_function, X, labels, num_samples=100, batch_size=50, reverse=True, device=torch.device('cpu')):
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
        batch_labels = labels[batch:batch+batch_size]
        padded_batch = pad_batch_input(batch_sentences, device=device)
        batch_scores = model.forward(padded_batch)
        acquisition_scores = acquisition_function(batch_scores)
        for j in range(0, len(batch_sentences)):
            input_sentence = batch_sentences[j]
            input_sentence = input_sentence.tolist()
            score = acquisition_scores[j].item()
            label = batch_labels[j]
            scored_samples.append((input_sentence, score, label))

    scored_samples.sort(reverse=reverse, key=lambda x: x[1])
    if num_samples > len(scored_samples):
        return samples, [], []
    else:
        samples = scored_samples[0:num_samples]
        unused_samples = scored_samples[num_samples: ]
        X_unlabeled = [torch.LongTensor(sentence) for sentence, score, label in unused_samples]
        labels = [label for sentence, score, label in unused_samples]
        return samples, X_unlabeled, labels


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
    accuracy = eval_network(dev, net, use_gpu=use_gpu, batch_size=batchSize, device=device)
    return total_loss, accuracy




def train_active_learning(net, vocab, X_seed, Y_seed, X_unlabeled, Y_gt, dev, num_epochs=15, human_label=False, acquisition_func=least_confidence, lr=0.001, batchSize=50, num_samples=100, use_gpu=False, device=torch.device('cpu')):
    """
    Main active learning training loop
    """
    print("Start Active Learning Training")
    optimizer = optim.Adam(net.parameters(), lr=lr)
    num_classes = len(set(Y_seed))
    epoch_losses = []
    eval_accuracy = []
    hand_labeled_data = []
    print("EPOCH: 0")
    total_loss, accuracy = train_step(net, X_seed, Y_seed, 0, dev, optimizer, num_classes=num_classes, batchSize=batchSize, use_gpu=use_gpu, device=device)
    epoch_losses.append(total_loss)
    eval_accuracy.append(accuracy)
    for epoch in range(1, num_epochs):
        print("EPOCH: " + str(epoch))
        # total_loss, accuracy = train_step(net, X_seed, Y_seed, epoch, dev, optimizer, num_classes=num_classes, batchSize=batchSize, use_gpu=use_gpu, device=device)
        # epoch_losses.append(total_loss)
        # eval_accuracy.append(accuracy)
        if len(X_unlabeled) > 0:
            samples_to_label, X_unlabeled, Y_gt = compute_acquisition_function(net, acquisition_func, X_unlabeled, Y_gt, num_samples=num_samples, batch_size=batchSize, device=device)
            new_labels = label_data(samples_to_label, vocab, use_human_labels=human_label)
            X_samples = [torch.LongTensor(sample) for sample, score, label in samples_to_label]
            for i in range(len(samples_to_label)):
                sample, score, label = samples_to_label[i]
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
        total_loss, accuracy = train_step(net, X_seed, Y_seed, epoch, dev, optimizer, num_classes=num_classes, batchSize=batchSize, use_gpu=use_gpu, device=device)
        epoch_losses.append(total_loss)
        eval_accuracy.append(accuracy)

    print("Finished Training")

    return epoch_losses, eval_accuracy, hand_labeled_data

def main():
    sampling_functions = ['random_score', 'entropy_score', 'least_confidence']
    sampling_sizes = [5000, 10000, 15000, 20000]
    num_active_samples = [10, 25, 50]

    args = parse_args()
    # twitter_csv_path = args.tweet_csv_file
    labeled_twitter_csv_path = args.labeled_tweet_csv_file
    unlabeled_twitter_csv_path = args.unlabeled_tweet_csv_file

    seed_data_size = args.seed_data_size
    use_bert = False
    shuffle = False
    train_data, dev_data, test_data = load_twitter_data(labeled_twitter_csv_path,
                                                        test_split_percent=0.1,
                                                        val_split_percent=0.2,
                                                        shuffle=shuffle,
                                                        overfit=True, use_bert=use_bert,
                                                        overfit_val=40000)
    unlabeled_tweets, ground_truth_labels = load_unlabeled_tweet_csv(unlabeled_twitter_csv_path, num_tweets=45000)
    X_unlabeled = train_data.convert_text_to_ids(unlabeled_tweets)
    ground_truth_labels = ground_truth_labels
    ground_truth_labels = (ground_truth_labels + 1.0)/2.0

    test_accuracies = {}

    print("Running ablation experiment on sampling functions and seed sizes")
    for af in sampling_functions:
        if af == 'random_score':
            acquisition_func = random_score
        elif af == 'entropy_score':
            acquisition_func = entropy_score
        elif af == 'least_confidence':
            acquisition_func = least_confidence
        for seed_data_size in sampling_sizes:
            for sample_size in num_active_samples:
                X_seed = train_data.Xwordlist[0:seed_data_size]
                Y_seed = train_data.labels[0:seed_data_size]
                Y_seed = (Y_seed + 1.0)/2.0
                cnn_net = CNN(train_data.vocab_size, DIM_EMB=300, NUM_CLASSES = 2)

                device = torch.device('cuda:0')
                cnn_net = cnn_net.cuda()
                print("Train active learning")
                epoch_losses, eval_accuracy, hand_labeled_data = train_active_learning(cnn_net, train_data,
                                                                    X_seed, Y_seed,
                                                                    X_unlabeled, np.copy(ground_truth_labels), dev_data,
                                                                    num_epochs=8, acquisition_func=acquisition_func,
                                                                    lr=0.0035, batchSize=150, num_samples=sample_size,
                                                                    use_gpu=True, device=device)
                cnn_net.eval()
                print("Test Set")
                test_accuracy = eval_network(test_data, cnn_net, use_gpu=True, device=device)
                param_combo = "CNN Active Learning: " + " Acquisition_Func: " + af + " Seed Size: " + str(seed_data_size) + " Sample Size: " + str(sample_size)
                test_accuracies[param_combo] = test_accuracy
                filename = "results_ablation/cnn_active_learning_val_accuracy_" + af + "_" + str(seed_data_size) + "_" + str(sample_size) + ".npy"
                np.save(filename, np.array(eval_accuracy))

    print("Finished experiments")
    with open("ablation_test_accuracies.txt", "w") as f:
        for key in test_accuracies.keys():
            accuracy = test_accuracies[key]
            line = key + " Acc: " + str(accuracy) + "\n"
            f.write(line)

if __name__ == '__main__':
    main()
