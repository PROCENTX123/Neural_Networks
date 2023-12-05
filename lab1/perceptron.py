import numpy as np

class Perceptron:
    def __init__(self, input_size, activation_function='linear', learning_rate=0.01, epochs=100):
        self.weights = np.zeros(input_size + 1)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.activation_function = self.get_activation_function(activation_function)

    def get_activation_function(self, name):
        if name == 'linear':
            return lambda x: x
        elif name == 'sigmoid':
            return lambda x: 1 / (1 + np.exp(-x))
        elif name == 'tan':
            return lambda x: np.tanh(x)
        elif name == 'relu':
            return lambda x: np.maximum(0, x)


    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return self.activation_function(summation)

    def train(self, training_inputs, labels):
        for epoch in range(self.epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                error = label - prediction
                self.weights[1:] += self.learning_rate * error * inputs
                self.weights[0] += self.learning_rate * error


def make_perceptron():
    perceptron_linear = Perceptron(input_size=25, activation_function='linear')
    perceptron_sigmoid = Perceptron(input_size=25, activation_function='sigmoid')
    perceptron_tan = Perceptron(input_size=25, activation_function='tan')
    perceptron_relu = Perceptron(input_size=25, activation_function='relu')

    return perceptron_linear, perceptron_sigmoid, perceptron_tan, perceptron_relu