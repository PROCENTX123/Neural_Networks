import numpy as np
from math_func import relu, der_relu, sigmoid, der_sigmoid, softmax
from main import LR

class NNLinear:
    wb: np.ndarray
    inputs: np.ndarray
    optimizer: str
    grad_pre: np.ndarray = None
    H: np.ndarray = None
    wb_pre: np.ndarray = None

    def __init__(self, in_len, out_len, optimizer, perceptron):
        self.wb = np.random.rand(in_len + 1, out_len) - 0.5
        self.optimizer = optimizer
        self.perceptron = perceptron

    def __repr__(self):
        return f'NNLinear({self.wb.shape})'

    def __eq__(self, o):
        return id(self) == id(o)

    def forward(self, inputs: np.ndarray):
        '''
        inputs - вектор-строка длинны in_len
        '''
        # print(f'    -> forward in {self}')
        self.inputs = np.append(inputs, [1])
        return self.inputs @ self.wb

    def simple_forward(self, inputs: np.ndarray):
        '''
        inputs - вектор-строка длинны in_len
        '''
        return np.append(inputs, [1]) @ self.wb

    def simple_backward(self, grad):
        return (grad @ self.wb.T)[:-1]

    def backward(self, grad):
        # print(f'BACKWARD IN {self}')
        # вычисляем градиент, который прокинем
        # до шага оптимизатора
        grad_to_ret = (grad @ self.wb.T)[:-1]  # отрезаем градиент для bias
        # считаем градиент относительно параметров слоя
        grad_wb = np.reshape(self.inputs, (-1, 1)) @ np.reshape(grad, (1, -1))
        # нормируем!!!
        if np.linalg.norm(grad_wb) != 0:
            grad_wb = grad_wb / np.linalg.norm(grad_wb)

        if np.linalg.norm(grad_to_ret) != 0:
            grad_to_ret = grad_to_ret / np.linalg.norm(grad_to_ret)

        if self.optimizer == 'GD':
            d = grad_wb
        elif self.optimizer == 'FR':
            # первый шаг
            if self.grad_pre is None:
                self.grad_pre = grad_wb
                b = 0
            # обычный шаг
            else:
                grad = grad_wb.flatten()
                grad_pre = self.grad_pre.flatten()
                if np.linalg.norm(grad_pre) > 0.00001:
                    b = (np.sum(grad * grad) / np.sum(grad_pre * grad_pre)) ** 2
                else:
                    b = 1
                b = max(min(b, 1), 0)
                # если предыдущий градиент был нулевой, то не учитываем его
                # if np.isnan(b) or b == np.Inf:
                #     b = 0
                # b = 0.1
            # считаем направление смещения
            d = (grad_wb + b * self.grad_pre)
            # запоминаем "старый" градиент
            self.grad_pre = grad_wb
        elif self.optimizer == 'BFGS':
            # https://github.com/kthohr/optim/blob/master/src/unconstrained/bfgs.cpp
            # инициализация
            if self.H is None:
                self.H = np.identity(len(grad_wb))
                self.grad_pre = np.zeros(grad_wb.shape)
                self.wb_pre = np.zeros(self.wb.shape)

            # считаем направление смещения
            d = (self.H @ grad_wb)

            # обновляем гессиан
            y = grad_wb - self.grad_pre
            s = self.wb - self.wb_pre

            W1 = np.eye(len(grad_wb)) - s @ y.T
            self.H = W1 @ self.H @ W1.T

            # запоминаем "старые" значения
            self.grad_pre = grad_wb
            self.wb_pre = self.wb
        else:
            raise RuntimeError('unknown optimizing method')

        # линейный поиск
        # нормируем d

        # print('start linear search')
        if np.linalg.norm(d) > 0.001:
            d = LR * d / np.linalg.norm(d)

        # self.wb -= d
        # return grad_to_ret

        def calc_special():
            return self.perceptron.calc_for_special_layer(self, self.inputs[:-1])

        old_res_val = calc_special()
        # print(f'    {old_res_val}')
        self.wb -= d
        new_res_val = calc_special()
        # print(f'    {new_res_val}')
        if new_res_val < old_res_val and abs(new_res_val - old_res_val) > 0.0001:
            while new_res_val < old_res_val and abs(new_res_val - old_res_val) > 0.0001:
                self.wb -= d
                old_res_val = new_res_val
                new_res_val = calc_special()
                # print(f'    {new_res_val}')
            self.wb += d

        # прокидываем градиент дальше
        return grad_to_ret
        # return self.perceptron.fit_special_layer_get_my_grad(self, self.inputs[:-1])


class NNRelu:
    inputs: np.array

    def forward(self, inputs):
        self.inputs = inputs
        return relu(inputs)

    def backward(self, grad):
        res = grad * der_relu(self.inputs)
        return res

class NNSigmoid:
    inputs: np.array

    def forward(self, inputs):
        self.inputs = inputs
        return sigmoid(inputs)

    def backward(self, grad):
        res = grad * der_sigmoid(self.inputs)
        return res


class NNSoftmax:
    inputs: np.ndarray

    def forward(self, inputs):
        self.inputs = inputs
        return softmax(inputs)

    def backward(self, grad):
        res = grad * softmax(self.inputs) * (1 - softmax(self.inputs))
        return res