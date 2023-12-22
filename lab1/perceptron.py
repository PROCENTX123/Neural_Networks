from activation_func import sigmoid, sigmoid_derivative, tanh, tanh_derivative, ReLu, ReLu_derivative


class Perceptron:
    def __init__(self, input_size, bias, epochs, activation_function, learning_rate=0.01):
        self.weights = [0] * input_size
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
            raise NotImplementedError(f'Activation {name} not implementation')

    def predict(self, inputs):
        summation = sum(x * w for x, w in zip(inputs, self.weights)) + self.bias
        return self.activation_function(summation)

    def train(self, training_inputs, labels):
        for epoch in range(self.epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                error = label - prediction
                self.weights += self.learning_rate * error * inputs
                self.bias += self.learning_rate * error
