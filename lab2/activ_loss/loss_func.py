import numpy as np


def mean_squared_error(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)


def cross_entropy(y_pred, y_true):
    return -np.sum(y_true * np.log(y_pred + 1e-9)) / y_pred.shape[0]


def kl_divergence():
    pass