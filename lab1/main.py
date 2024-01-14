import numpy as np
from tqdm import tqdm
from data import training_data, test_data
from perceptron import Perceptron
from const import ACTIVATION_NAMES, EPOCH, accuracy, loss
import matplotlib.pyplot as plt


def calc_accuracy(training_labels, pred):
    correct = 0
    for i, label in enumerate(training_labels):
        prediction = np.round(pred[i]).astype(int)
        correct += np.array_equal(label, prediction)
    accuracy = correct / len(training_labels)
    return accuracy



def calc_loss(training_labels, pred):
    return np.mean((training_labels - pred) ** 2)


if __name__ == "__main__":
    training_inputs, training_labels = training_data()
    test_inputs = test_data()
    test_labels = training_labels

    for activation_name in ACTIVATION_NAMES:
        accuracy[activation_name] = []
        loss[activation_name] = []

        for epoch in tqdm(EPOCH):
            perceptron = Perceptron(input_size=len(training_inputs[0]), epochs=epoch, activation_function=activation_name)

            perceptron.train(training_inputs, training_labels)

            pred = [perceptron.predict(inputs) for inputs in test_inputs]

            accuracy[activation_name].append(calc_accuracy(test_labels, pred))
            loss[activation_name].append(calc_loss(test_labels, pred))

    fig, axs = plt.subplots(nrows=2, figsize=(15, 15))
    for name in ACTIVATION_NAMES:
        axs[0].plot(EPOCH, accuracy[name], label=name)
        axs[1].plot(EPOCH, loss[name], label=name)

    axs[0].set(ylabel='Accuracy', xlabel='Epochs', title='Accuracy per Epoch')
    axs[1].set(ylabel='Loss', xlabel='Epochs', title='Loss per Epoch')

    for ax in axs:
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig('accuracy_loss_plots100epochmax.png')
    plt.show()