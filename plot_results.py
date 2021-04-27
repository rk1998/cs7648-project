import numpy as np
import matplotlib.pyplot as plt
import pandas as pd




def plot_results_from_csv(csv_file_path, model_name=""):
    results = pd.read_csv(csv_file_path)
    results = results.fillna(0.0)
    validation_accuracy = results['val_acc'].values
    validation_accuracy = validation_accuracy[validation_accuracy != 0]
    training_loss = results['train_loss'].values
    batch_size = int(training_loss.shape[0]/5)
    training_loss = training_loss[0:training_loss.shape[0]:batch_size]
    plt.figure()
    plt.plot(validation_accuracy, 'ro-')
    plt.title("Twitter Dataset Accuracy: " + model_name)
    plt.xlabel("Epochs")
    plt.ylabel("Validation Accuracy")
    plt.show()

    plt.figure(1)
    plt.plot(training_loss, 'b-')
    plt.title("Twitter Dataset Training Loss: " + model_name)
    plt.xlabel("Training Iteration")
    plt.ylabel("Loss")
    plt.show()



def plot_accuracy(accuracy_results, model_name, training_set_size):
    # min_accs, accs, max_accs = accuracy_results
    plt.figure()
    plt.plot(accuracy_results, 'ro-')
    # plt.plot(min_accs, 'bo-', label="min_accuracy")
    # plt.plot(max_accs, 'go-', label="max_accuracy")
    plt.title("Twitter Dataset Accuracy: " + model_name + " ,Training Set Size: " + str(training_set_size))
    plt.xlabel("Epochs")
    plt.ylabel("Validation Accuracy")
    # plt.legend()
    plt.show()

def plot_losses(loss_results, model_name, training_set_size):
    plt.figure()
    plt.plot(loss_results, 'r-')
    plt.title("Twitter Dataset Loss (CE): " + model_name + " ,Training Set Size: " + str(training_set_size))
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()


def compare_accuracy_results(results_1, results_2, label1="baseline", label2="AL (Least Confidence)"):
    plt.figure()
    plt.plot(results_1, "ro-", label=label1)
    plt.plot(results_2, "bo-", label=label2)
    plt.title("Twitter Dataset Accuracy Comparison")
    plt.xlabel("Epochs")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.show()


def compare_multiple_results(results_list, labels, colors, title="Twitter Dataset Accuracy Comparison"):
    plt.figure()
    for i in range(len(results_list)):
        result = results_list[i]
        label = labels[i]
        color_tag = colors[i]
        plt.plot(result, color_tag, label=label)

    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.show()



def main():
    result1_file = "cnn_active_learning_validation_accuracyleast_confidence_9000.npy"
    result2_file = "cnn_active_learning_validation_accuracyrandom_9000.npy"
    result1 = np.load(result1_file)
    result2 = np.load(result2_file)
    compare_accuracy_results(result1, result2, label1="CNN AL (Least Confidence)", label2 = "CNN AL (Random)")
    # csv_file_path = "metrics.csv"
    # model_name = "LSTM + BERT"
    # plot_results_from_csv(csv_file_path, model_name=model_name)

if __name__ == '__main__':
    main()
