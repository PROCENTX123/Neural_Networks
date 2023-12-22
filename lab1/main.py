import numpy as np
from tqdm import tqdm
from data import training_data, test_data
from perceptron import Perceptron
from const import ACTIVATION_NAMES, EPOCH, accuracy, loss
import matplotlib.pyplot as plt


def calc_accuracy(training_labels, pred):
    correct = 0
    for i in range(len(training_labels)):
        # if training_labels[i] == pred[i]:
        if abs(training_labels[i] - pred[i]) < 0.15:
            correct += 1
    accuracy = correct / len(training_labels)

    return accuracy


def calc_loss(training_labels, pred):
    return np.mean((training_labels - pred) ** 2)


if __name__ == "__main__":

    training_inputs, training_labels = training_data()

    test_inputs = test_data()


    for activation_name in ACTIVATION_NAMES:
        for epoch in tqdm(EPOCH):
            perceptron = Perceptron(input_size=25, bias=0, epochs=epoch, activation_function=activation_name)

            perceptron.train(training_inputs, training_labels)

            pred = [perceptron.predict(inputs) for inputs in test_inputs]

            accuracy[activation_name].append(calc_accuracy(training_labels, pred))
            loss[activation_name].append(calc_loss(training_labels, pred))

    fig, axs = plt.subplots(nrows=2, figsize=(15, 15))
    for name in ACTIVATION_NAMES:
        axs[0].plot(accuracy[name])
        axs[0].legend(ACTIVATION_NAMES)
        axs[0].set(ylabel='Accuracy')
        axs[0].set(xlabel='Epochs')

        axs[1].plot(loss[name])
        axs[1].legend(ACTIVATION_NAMES)
        axs[1].set(ylabel='Loss')
        axs[1].set(xlabel='Epochs')

    plt.tight_layout()
    plt.savefig('accuracy_loss_plots.png')
    plt.show()
