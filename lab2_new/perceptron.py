from math_func import LossFunc
import numpy as np
from neural_networks_layers import NNLinear, NNSigmoid, NNSoftmax

class Perceptron:
    layers: list
    loss: LossFunc
    last_target: np.ndarray

    def __init__(self, sizes: list[int], loss, optimizer):
        self.layers = [
            x for i, o in zip(sizes, sizes[1:]+[10]) for x in [NNLinear(i, o, optimizer, self), NNSigmoid()]
        ] + [NNSoftmax()]
        self.loss = loss

    def forward(self, inputs):
        res = inputs
        for layer in self.layers:
            res = layer.forward(res)
        return res

    def backward(self, target, pred):
        grad_by_last_layer_outs = self.loss.df(target, pred)
        grad = grad_by_last_layer_outs
        for layer in self.layers[::-1]:
            grad = layer.backward(grad)

    def fit(self, x, y):
        self.last_target = y
        pred = self.forward(x)
        self.backward(y, pred)
        return np.argmax(pred)

    def predict(self, x, y):
        pred = self.forward(x)
        return np.argmax(pred)

    def calc_for_special_layer(self, layer: NNLinear, inputs: np.ndarray):
        index = self.layers.index(layer)
        res = inputs
        for layer in self.layers[index:]:
            res = layer.forward(res) if type(
                layer) != NNLinear else layer.simple_forward(res)
        return self.loss.f(self.last_target, res)

    def fit_special_layer_get_my_grad(self, layer: NNLinear, inputs: np.ndarray):
        index = self.layers.index(layer)
        res = inputs
        for layer in self.layers[index:]:
            res = layer.forward(res) if type(
                layer) != NNLinear else layer.simple_forward(res)
        grad_by_last_layer_outs = self.loss.df(self.last_target, res)
        grad = grad_by_last_layer_outs
        for layer in self.layers[index:][::-1]:
            grad = layer.backward(grad) if type(
                layer) != NNLinear else layer.simple_backward(res)
        return grad
