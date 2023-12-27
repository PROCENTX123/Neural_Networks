import numpy as np
from collections import namedtuple
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt


# Loss Functions
LossFunc = namedtuple('LossFunc', ['f', 'df', 'name'])

def mse(x, y): return np.mean((x-y)**2)
def der_mse(x, y): return (x-y) * -2 / x.size

def cross_entropy(target, pred):
    pred = np.clip(pred, 1e-8, 1 - 1e-8)
    return -np.mean(target * np.log(pred))

def der_cross_entropy(target, pred):
    pred = np.clip(pred, 1e-8, 1 - 1e-8)
    return pred - target

def kl(target, pred):
    pred = np.clip(pred, 1e-10, 1 - 1e-10)
    target = np.clip(target, 1e-10, 1 - 1e-10)
    return np.mean(target * np.log(target / pred))

def der_kl(target, pred):
    pred = np.clip(pred, 1e-8, 1 - 1e-8)
    return pred - target

# Activation Functions
def relu(x): return np.maximum(0, x)
def der_relu(x): return (x > 0).astype(float)
def sigmoid(x): return 1 / (1 + np.exp(-x))
def der_sigmoid(x): sig = sigmoid(x); return sig * (1 - sig)
def softmax(x): exps = np.exp(x - np.max(x)); return exps / np.sum(exps, axis=0)

# Loss Function Instances
MSE = LossFunc(mse, der_mse, 'MSE LOSS')
CrossEntropy = LossFunc(cross_entropy, der_cross_entropy, 'Cross Entropy LOSS')
KLDiv = LossFunc(kl, der_kl, 'Kullback–Leibler Divergence LOSS')

# Base Neural Network Layer Class
class NNLayer:
    def forward(self, inputs: np.ndarray):
        raise NotImplementedError

    def backward(self, grad: np.ndarray):
        raise NotImplementedError

# NNLinear Layer
class NNLinear(NNLayer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros(output_size)
        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        return np.dot(inputs, self.weights) + self.bias

    def backward(self, grad):
        grad_input = np.dot(grad, self.weights.T)
        grad_weights = np.dot(self.inputs.T, grad)
        grad_bias = np.sum(grad, axis=0)
        # Update weights and biases here or return gradients for later update
        return grad_input

# NNRelu Layer
class NNRelu(NNLayer):
    def forward(self, inputs):
        self.inputs = inputs
        return relu(inputs)

    def backward(self, grad):
        return grad * der_relu(self.inputs)

# NNSigmoid Layer
class NNSigmoid(NNLayer):
    def forward(self, inputs):
        self.inputs = inputs
        return sigmoid(inputs)

    def backward(self, grad):
        return grad * der_sigmoid(self.inputs)

# NNSoftmax Layer
class NNSoftmax(NNLayer):
    def forward(self, inputs):
        self.inputs = inputs
        return softmax(inputs)

    def backward(self, grad):
        return grad  # Softmax gradient is handled in the loss function

# Perceptron Class
class Perceptron:
    def __init__(self, layer_sizes, loss_func):
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(NNLinear(layer_sizes[i], layer_sizes[i+1]))
            self.layers.append(NNSigmoid() if i < len(layer_sizes) - 2 else NNSoftmax())
        self.loss = loss_func

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def train(self, x, y):
        output = self.forward(x)
        loss = self.loss.f(y, output)
        grad = self.loss.df(y, output)
        self.backward(grad)
        return loss

# Data Loading and Preprocessing
ds = datasets.MNIST(root='data', train=True, download=True,
                    transform=lambda img: np.array(img).flatten() / 255.0,
                    target_transform=lambda x: np.eye(10)[x])
ds_train, ds_val = Subset(ds, range(50000)), Subset(ds, range(50000, 60000))
dl_train, dl_val = DataLoader(ds_train, batch_size=32), DataLoader(ds_val, batch_size=32)

# Training Loop
def train_network(network, train_loader, val_loader=None, epochs=10):
    training_loss = []
    validation_accuracy = []

    for epoch in range(epochs):
        total_loss = 0
        for x, y in train_loader:
            x = np.array(x)  # Convert x and y to numpy arrays
            y = np.array(y)
            loss = network.train(x, y)
            total_loss += loss
        avg_loss = total_loss / len(train_loader)
        training_loss.append(avg_loss)

        # Calculate validation accuracy if validation data is provided
        if val_loader:
            correct = 0
            total = 0
            for x, y in val_loader:
                x = np.array(x)
                y = np.array(y)
                predictions = network.forward(x)
                predicted_labels = np.argmax(predictions, axis=1)
                true_labels = np.argmax(y, axis=1)
                correct += (predicted_labels == true_labels).sum()
                total += y.shape[0]
            accuracy = correct / total
            validation_accuracy.append(accuracy)
            print(f"Epoch {epoch+1}, Loss: {avg_loss}, Validation Accuracy: {accuracy}")
        else:
            print(f"Epoch {epoch+1}, Loss: {avg_loss}")

    # Plotting
    plt.figure(figsize=(12, 5))

    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(training_loss, label='Training Loss')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot validation accuracy
    if val_loader:
        plt.subplot(1, 2, 2)
        plt.plot(validation_accuracy, label='Validation Accuracy')
        plt.title('Validation Accuracy per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

    plt.show()

# Create and Train the Network
network = Perceptron([784, 64, 10], CrossEntropy)
train_network(network, dl_train, dl_val, epochs=100)