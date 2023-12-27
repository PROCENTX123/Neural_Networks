import numpy as np
from lab2.activ_loss.activation_func import ReLu, ReLu_derivative, SoftMax, SoftMax_derivative
from lab2.activ_loss.loss_func import mean_squared_error, cross_entropy, kl_divergence


class MLP:
    def __init__(self, layer_sizes, epochs, activation_function, learning_rate, loss_func, optim_method):
        self.weights = [np.random.randn(x, y) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases = [np.random.randn(1, y) for y in layer_sizes[1:]]
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.activation_function, self.derivative_function = self.get_activation_func(activation_function)
        self.loss_func = self.get_loss_func(loss_func)
        self.optim_method = optim_method

        self.prev_grad_w = None
        self.prev_grad_b = None
        self.hessian_inv = None

    def get_activation_func(self, name):
        match name:
            case "ReLu":
                return ReLu, ReLu_derivative
            case "SoftMax":
                return SoftMax, SoftMax_derivative
            case _:
                raise NotImplementedError(f'Activation {name} not implementation')

    def get_loss_func(self, name):
        match name:
            case "MSE":
                return mean_squared_error
            case "Cross entropy":
                return cross_entropy
            case "KL":
                return kl_divergence
            case _:
                raise NotImplementedError(f'Loss {name} not implementation')


    def forward(self, x):
        a = x
        for i in range(len(self.weights)):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            a = self.activation_function(z)
        return a

    def backpropagate(self, x, y):
        grad_w = [np.zeros(w.shape) for w in self.weights]
        grad_b = [np.zeros(b.shape) for b in self.biases]


        activation = x
        activations = [x]
        zs = []

        for w, b in zip(self.weights, self.biases):
            z = np.dot(activation, w) + b
            zs.append(z)
            activation = self.activation_function(z)
            activations.append(activation)

        # Backward pass
        delta = self.loss_func(activations[-1], y) * self.derivative_function(zs[-1])
        grad_w[-1] = np.dot(activations[-2].T, delta)
        grad_b[-1] = np.sum(delta, axis=0)

        for l in range(2, len(self.weights) + 1):
            z = zs[-l]
            sp = self.derivative_function(z)
            delta = np.dot(delta, self.weights[-l + 1].T) * sp
            grad_w[-l] = np.dot(activations[-l - 1].T, delta)
            grad_b[-l] = np.sum(delta, axis=0)

        return grad_w, grad_b

    def train(self, x_train, y_train):
        for epoch in range(self.epochs):
            grad_w, grad_b = self.backpropagate(x_train, y_train)

            if self.optim_method == "Gradient Descent":
                for i in range(len(self.weights)):
                    self.weights[i] -= self.learning_rate * grad_w[i]
                    self.biases[i] -= self.learning_rate * grad_b[i]



            elif self.optim_method == "Fletcher-Reeves":
                if self.prev_grad_w is None or self.prev_grad_b is None:
                    beta = 0
                else:
                    numerator = sum([np.sum(np.square(gw)) for gw in grad_w])
                    denominator = sum([np.sum(np.square(pg)) for pg in self.prev_grad_w])
                    beta = numerator / denominator

                new_grad_w = []
                new_grad_b = []
                for i in range(len(self.weights)):
                    new_gw = grad_w[i] + beta * (self.prev_grad_w[i] if self.prev_grad_w else 0)
                    new_gb = grad_b[i] + beta * (self.prev_grad_b[i] if self.prev_grad_b else 0)
                    self.weights[i] -= self.learning_rate * new_gw
                    self.biases[i] -= self.learning_rate * new_gb
                    new_grad_w.append(new_gw)
                    new_grad_b.append(new_gb)

                self.prev_grad_w = new_grad_w
                self.prev_grad_b = new_grad_b



            elif self.optim_method == "BFGS":
                if self.hessian_inv is None:
                    self.hessian_inv = [np.eye(w.shape[0]) for w in self.weights]

                for i in range(len(self.weights)):
                    s = -self.learning_rate * grad_w[i]
                    self.weights[i] += s
                    self.biases[i] -= self.learning_rate * grad_b[i]

                    y = grad_w[i] - (self.prev_grad_w[i] if self.prev_grad_w else np.zeros_like(grad_w[i]))
                    rho = 1.0 / np.dot(y, s)

                    I = np.eye(len(s))
                    term1 = I - rho * np.outer(s, y)
                    term2 = I - rho * np.outer(y, s)
                    self.hessian_inv[i] = np.dot(np.dot(term1, self.hessian_inv[i]), term2) + rho * np.outer(s, s)

                self.prev_grad_w = grad_w
                self.prev_grad_b = grad_b

