from activation_func import sigmoid, sigmoid_derivative, tanh, tanh_derivative, ReLu, ReLu_derivative
import numpy as np

class Perceptron:
    def __init__(self, input_size, bias, epochs, activation_function, learning_rate=0.01):
        # Initialize weights with small random values instead of zeros to break symmetry
        self.weights = np.random.randn(input_size) * 0.01
        self.bias = bias
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.activation_function, self.derivative_function = self.get_activation_function(activation_function)

    def get_activation_function(self, name):
        if name == 'sigmoid':
            return sigmoid, sigmoid_derivative
        elif name == 'tan':
            return tanh, tanh_derivative
        elif name == 'relu':
            return ReLu, ReLu_derivative
        else:
            raise ValueError(f'Activation function {name} is not implemented')

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights) + self.bias
        return self.activation_function(summation)

    def train(self, training_inputs, labels):
        for epoch in range(self.epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                error = label - prediction

                self.weights += self.learning_rate * error * inputs * self.derivative_function(prediction)
                self.bias += self.learning_rate * error
