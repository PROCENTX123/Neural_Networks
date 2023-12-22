import numpy as np
from tqdm import tqdm
from data import training_data, test_data
from perceptron import Perceptron
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


ACTIVATION_NAMES = ['sigmoid', 'tan', 'relu']
EPOCH = [10, 100, 200,  300, 400, 500, 600, 700, 800, 900, 1000]
accuracy = {
    'sigmoid': [],
    'tan': [],
    'relu': []
}

loss = {
    'sigmoid': [],
    'tan': [],
    'relu': []
}


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

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    for activation_name in ACTIVATION_NAMES:
        plt.plot(EPOCH, accuracy[activation_name], label=activation_name)
    plt.title('Accuracy vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    for activation_name in ACTIVATION_NAMES:
        plt.plot(EPOCH, loss[activation_name], label=activation_name)
    plt.title('Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('accuracy_loss_plots.png')
    plt.show()
