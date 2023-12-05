import numpy as np


def training_data():
    data = {
        '0': np.array([[0, 1, 1, 1, 0],
                       [1, 0, 0, 0, 1],
                       [1, 0, 0, 0, 1],
                       [1, 0, 0, 0, 1],
                       [0, 1, 1, 1, 0]]),

        '1': np.array([[0, 0, 1, 0, 0],
                       [0, 1, 1, 0, 0],
                       [0, 0, 1, 0, 0],
                       [0, 0, 1, 0, 0],
                       [0, 1, 1, 1, 0]]),



        'А': np.array([[0, 1, 1, 1, 0],
                       [1, 0, 0, 0, 1],
                       [1, 1, 1, 1, 1],
                       [1, 0, 0, 0, 1],
                       [1, 0, 0, 0, 1]]),

    }

    training_data = []
    labels = []

    for label, matrix in data.items():
        flattened_matrix = matrix.flatten()
        training_data.append(flattened_matrix)
        labels.append(int(label.isdigit()))

    return np.array(training_data), np.array(labels)


def test_data():

    test_data = {

        '0': np.array([[0, 1, 1, 1, 0],
                       [1, 0, 0, 0, 1],
                       [1, 1, 1, 1, 1],
                       [1, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0]]),

        '1': np.array([[0, 1, 0, 0, 0],
                       [1, 1, 0, 0, 0],
                       [0, 1, 0, 0, 0],
                       [0, 1, 0, 0, 0],
                       [1, 1, 1, 0, 0]]),

        'А': np.array([[0, 1, 1, 1, 0],
                       [1, 0, 0, 0, 1],
                       [1, 1, 1, 1, 1],
                       [1, 0, 0, 0, 1],
                       [1, 0, 0, 0, 1]])

    }

    test_inputs = []
    test_labels = []

    for label, matrix in test_data.items():
        flattened_matrix = matrix.flatten()
        test_inputs.append(flattened_matrix)
        test_labels.append(int(label.isdigit()))

    return np.array(test_inputs), np.array(test_labels)