from multiprocessing.sharedctypes import Value
from typing import List, Callable
import numpy as np
from layers import Layer
from loss_functions import *


class Sequential:

    def __init__(self, layers: List[Layer], loss_function: Callable):
        self.layers = layers
        self.loss_function = loss_function
        # self.epochs = epochs
        # self.learning_rate = learning_rate

        self._check_valid_layers()

    def _check_valid_layers(self):
        if not self.layers:
            raise ValueError(
                "layers argument is empty. should contain atleast one layer")
        for layer in self.layers:
            if not isinstance(layer, Layer):
                raise TypeError(
                    "layers argument should be a list of classes that inherit from Layer class")
        if len(self.layers) > 1:
            for fan_out, fan_in in zip(self.layers[:-1], self.layers[1:]):
                if fan_out.dimensions[1] != fan_in.dimensions[0]:
                    raise ValueError("amount of output neurons of every layer should " +
                                     "correspond to amount of input neurons in next layer")

    def fit(self, X, Y, epochs: int = 1000, learning_rate: float = 0.01, stochastic: bool = True, verbose: bool = True):
        if not Y.shape[1] == self.layers[-1].dimensions[1]:
            raise ValueError(
                f"Y size ({Y.shape[1]}) should be equal to output layer size ({self.layers[-1].dimensions[1]})")
        if stochastic:
            self._stochastic_gradient_descent(X, Y, epochs, learning_rate, verbose)
        else:
            self._gradient_descent(X, Y, epochs, learning_rate, verbose)

    def _gradient_descent(self, X, Y, epochs, learning_rate, verbose):
        error = []

        X = X.T
        # Y = Y.reshape(-1, Y.shape[1])
        Y = Y.reshape(Y.shape[0], -1).T
        print(Y.shape)

        for epoch in range(epochs):
            output = self._forward_propagation(X)

            # print("model output:", output)
            # Y must be one-hot encoded first, so that it matches output
            error.append(self.loss_function.forward(output, Y))

            gradient = self.loss_function.backward(output, Y)
            # print("loss:", error[-1])
            # print("model loss gradient:", gradient)
            self._backward_propagation(gradient, learning_rate)
            if verbose and (epoch + 1) % 50 == 0:
                print(f"epoch: {epoch + 1}/{epochs}, error={error[-1]}")
        # print(error)
        import matplotlib.pyplot as plt
        plt.plot(range(len(error)), error, 'r')
        plt.show()

    def _stochastic_gradient_descent(self, X, Y, epochs, learning_rate, verbose):
        error = []
        for epoch in range(epochs):
            for i, sample in enumerate(X):
                sample = sample.reshape(-1, 1)
                output = self._forward_propagation(sample)

                # print("model output:", output)
                # Y must be one-hot encoded first, so that it matches output
                error.append(self.loss_function.forward(output, Y[i].reshape(-1, 1)))

                gradient = self.loss_function.backward(output, Y[i].reshape(-1, 1))
                print("model loss gradient:", gradient)
                self._backward_propagation(gradient, learning_rate)
                if verbose and (epoch + 1) % 50 == 0:
                    print(f"epoch: {epoch + 1}/{epochs}, error={error[-1]}")
        # print(error)
        import matplotlib.pyplot as plt
        plt.plot(range(len(error)), error, 'r')
        plt.show()

    def _forward_propagation(self, X) -> np.array:
        output = X
        for layer in self.layers:
            output = layer.forward(output)
            # print("layer output: ", output)
        return output

    def _backward_propagation(self, gradient, learning_rate) -> np.array:
        for i, layer in enumerate(reversed(self.layers)):
            # print("layer from back to front:", i)
            gradient = layer.backward(gradient, learning_rate)
