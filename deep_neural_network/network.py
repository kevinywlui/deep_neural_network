import numpy as np
import math

from typing import List


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


class Network:
    def __init__(
        self, input_size: int, layer_sizes: List[int], activation_functions: List[str]
    ) -> None:
        self.nlayers = len(layer_sizes)
        self.input_size = input_size
        self.activation_functions = activation_functions

        # Initialize weights and biases
        self.biases = [np.random.randn(x, 1) for x in layer_sizes]
        self.weights = [
            np.random.randn(x, y)
            for x, y in zip(layer_sizes, [input_size] + layer_sizes[:-1])
        ]

    def _init_activation(self, activation_functions: List[str]) -> None:
        """Initialize lists of activation functions and their derivatives.
        """
        self._af: List[float] = []
        self._af_der: List[float] = []
        for func_name in activation_functions:
            if func_name == "sigmoid":
                self._af.append(sigmoid)
                self._af_der.append(sigmoid_prime)
            else:
                raise NotImplementedError(f"{func_name} is not implemented")

    def _forward_propagate(self, X):
        """Return the output of this network if the input is ``X``.       
        """
        As = [X]
        Zs = [None]
        for l in range(self.nlayers):
            Zs.append(np.dot(self.weights[l], As[-1]) + self.biases[l])
            As.append(self.activation_functions[l](Z[-1]))
        return As, Zs

    def evaluate(self, X):
        return self._forward_propagate(X)[0][-1]

    def _backward_propagate(self, X, Y):
        """Return the gradient.
        """
        # Forward propagate
        As, Zs = self._forward_propagate(X)

        # Initialize gradients
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]

        # Compute gradients by backward propagation

        # Base case, last layer
        dZ = 2 * (As[-1], Y) * self._af_der[-1](Zs[-1])
        grad_b[-1] = dZ
        grad_w[-1] = dZ * np.dot(delta, As[-2].transpose())

        # Recursively determine the rest
        for l in range(self.nlayers - 2, 0, -1):
            dA = np.dot(self.weights[l + 1].transpose(), dZ)
            dZ = dA * self._af_der[l](Zs[l])

            grad_b[l] = dZ
            grad_w[l] = np.dot(dZ, As[l - 1].transpose())

        return grad_b, grad_w

    def train(
        self, training_data, batch_size: int, n_epoch: int, learning_rate: float
    ) -> None:
        n = len(training_data)

        for epoch in range(n_epoch):

            # shuffle training data before each epoch
            np.random.shuffle(training_data)

            # Break into batches
            n_batches = math.ceil(n / batch_size)
            batches = [
                training_data[k, k + batch_size] for k in range(0, n, batch_size)
            ]

            # Process each batch
            for batch in batches:
                self._update_batch(batch, learning_rate)

    def _update_batch(self, batch, learning_rate):
        """Update the parameters.
        """
        # Initialize the gradients
        grad_b_acc = [np.zeros(b.shape) for b in self.biases]
        grad_w_acc = [np.zeros(w.shape) for w in self.weights]

        # Loop through batch and accumulate the gradients
        for X, Y in batch:
            grad_b, grad_w = self._backward_propagate(X, Y)
            grad_b_acc += grad_b
            grad_w_acc += grad_w

        m = len(batch)
        self.biases -= (learning_rate / m) * grad_b_acc
        self.weight -= (learning_rate / m) * grad_w_acc
