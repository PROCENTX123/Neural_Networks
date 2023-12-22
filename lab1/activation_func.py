import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
def tanh_derivative(x):
    return 1 - tanh(x)**2


def ReLu(x):
    return x * (x > 0)

def ReLu_derivative(x):
    return 1. * (x > 0)
