import numpy as np


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Инициализация весов сети
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)

        # Инициализация смещений
        self.bias_hidden = np.zeros((1, hidden_size))
        self.bias_output = np.zeros((1, output_size))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def softmax(self, x):
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def sigmoid(self, x):
        # Функция активации (сигмоид)
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        # Производная функции активации (сигмоид)
        return x * (1 - x)



    def feedforward(self, inputs):
        # Прямое распространение
        hidden = self.sigmoid(np.dot(inputs, self.weights_input_hidden) + self.bias_hidden)
        output = self.sigmoid(np.dot(hidden, self.weights_hidden_output) + self.bias_output)
        return output

    def train(self, inputs, targets, learning_rate=0.1, epochs=10000):
        for epoch in range(epochs):
            # Прямое распространение
            hidden = self.sigmoid(np.dot(inputs, self.weights_input_hidden) + self.bias_hidden)
            output = self.sigmoid(np.dot(hidden, self.weights_hidden_output) + self.bias_output)

            # Ошибка
            output_error = targets - output
            hidden_error = output_error.dot(self.weights_hidden_output.T)

            # Обратное распространение ошибки
            output_delta = output_error * self.sigmoid_derivative(output)
            hidden_delta = hidden_error * self.sigmoid_derivative(hidden)

            # Обновление весов и смещений
            self.weights_hidden_output += hidden.T.dot(output_delta) * learning_rate
            self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
            self.weights_input_hidden += inputs.T.dot(hidden_delta) * learning_rate
            self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

        print("Обучение завершено.")