import numpy as np

# Activation Functions
def relu(x):
    return np.maximum(0, x)

def der_relu(x):
    return np.where(x > 0, 1, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def der_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

def softmax(x):
    if np.linalg.norm(x) < 0.001:
        return np.zeros(len(x))
    x = x / np.linalg.norm(x)
    return np.exp(x) / np.sum(np.exp(x)) if np.sum(np.exp(x)) > 0.01 else np.zeros(len(x))

# Loss Functions
class LossFunc:
    def __init__(self, func, der_func, name):
        self.f = func
        self.df = der_func
        self.name = name

def mse(x, y):
    return np.sum((x - y) ** 2) / len(x)

def der_mse(x, y):
    res = (x - y) * 2 / len(x) * -1
    return res / np.sum(np.abs(res))

MSE = LossFunc(mse, der_mse, 'MSE LOSS')

def cross_entropy(target, pred):
    pred = np.clip(pred, 1e-8, 1 - 1e-8)
    return -np.mean(target * np.log(pred))

def der_cross_entropy(target, pred):
    pred = np.clip(pred, 1e-8, 1 - 1e-8)
    res = pred - target
    return res / np.linalg.norm(res)

CrossEntropy = LossFunc(cross_entropy, der_cross_entropy, 'Cross Entropy LOSS')


def kl(target, pred):
    pred = np.clip(pred, 1e-10, 1 - 1e-10)
    target = np.clip(pred, 1e-10, 1 - 1e-10)
    res = np.mean(target * np.log(target / pred))
    return res

def der_kl(target, pred):
    pred = np.clip(pred, 1e-8, 1 - 1e-8)
    res = pred - target
    return res / np.linalg.norm(res)

KLDiv = LossFunc(kl, der_kl, 'Kullback–Leibler Divergence LOSS')