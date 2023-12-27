import numpy as np


def SoftMax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def SoftMax_derivative(SoftMax_output):
    s = SoftMax_output.rehsape(-1, 1)
    return np.diag(s) - np.dot(s, s.T)


def ReLu(x):
    return np.maximum(0, x)


def ReLu_derivative(x):
    return np.where(x > 0, 1, 0)