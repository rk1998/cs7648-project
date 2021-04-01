import numpy as np
import matplotlib.pyplot as plt

def plot_accuracy(accuracy_results, model_name, training_set_size):
    min_accs, accs, max_accs = accuracy_results
    plt.figure()
    plt.plot(accs, 'ro-')
    plt.fill_between(list(range(len(accs))), min_accs, max_accs, alpha=0.3, color='r')
    plt.title("Twitter Dataset Accuracy: " + model_name + " ,Training Set Size: " + str(training_set_size))
    plt.xlabel("Epochs")
    plt.ylabel("Validation Accuracy")
    plt.show()

def plot_losses(loss_results, model_name, training_set_size):
    plt.figure()
    plt.plot(loss_results, 'r-')
    plt.title("Twitter Dataset Loss (CE): " + model_name + " ,Training Set Size: " + str(training_set_size))
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()
