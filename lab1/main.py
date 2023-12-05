from data import training_data, test_data
from perceptron import make_perceptron


def train(training_inputs, training_labels):
    perceptron_linear.train(training_inputs, training_labels)
    perceptron_sigmoid.train(training_inputs, training_labels)
    perceptron_tan.train(training_inputs, training_labels)
    perceptron_relu.train(training_inputs, training_labels)


if __name__ == "__main__":

    perceptron_linear, perceptron_sigmoid, perceptron_tan, perceptron_relu = make_perceptron()

    #training_data
    training_inputs, training_labels = training_data()

    #test_data
    test_inputs, test_labels = test_data()

    # Train perceptrons
    train(training_inputs, training_labels)

    # Test predictions
    predictions_linear = [perceptron_linear.predict(inputs) for inputs in test_inputs]
    predictions_sigmoid = [perceptron_sigmoid.predict(inputs) for inputs in test_inputs]
    predictions_tan = [perceptron_tan.predict(inputs) for inputs in test_inputs]
    predictions_relu = [perceptron_relu.predict(inputs) for inputs in test_inputs]

    print("Linear Activation:", predictions_linear)
    print("Sigmoid Activation:", predictions_sigmoid)
    print("Tanh Activation:", predictions_tan)
    print("ReLU Activation:", predictions_relu)
