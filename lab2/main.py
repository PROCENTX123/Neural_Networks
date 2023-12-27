import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from MLP import MLP
from data import
from const import ACTIVATION_NAMES, EPOCH

# from keras.utils import to_categorical



if __name__ == "__main__":
    #
    # for activation_name in ACTIVATION_NAMES:
    #
    #     for epoch in tqdm(EPOCH):
    #         perceptron = MLP(
    #             layer_sizes=[],
    #             epochs=epoch,
    #             activation_function=activation_name,
    #             learning_rate=0.01,
    #             loss_func="MSE",
    #             optim_method="GD"
    #         )
    #
    #         perceptron.train(X_train, y_train)
    #
    #         pred = perceptron.forward(X_test)

    perceptron = MLP(
        layer_sizes=[784, 64, 10],
        epochs=100,
        activation_function="ReLu",
        learning_rate=0.01,
        loss_func="MSE",
        optim_method="GD"
    )
    # y_train_one_hot = to_categorical(y_train)
    # y_test_one_hot = to_categorical(y_test)

    perceptron.train(X_train, y_train)

    pred = perceptron.forward(X_test)

    print(pred)