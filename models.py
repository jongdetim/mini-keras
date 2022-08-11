from typing import List, Callable
import numpy as np
from layers import Layer
from loss_functions import *

class Sequential:

    def __init__(self, layers: List[Layer], loss_function: Loss):
        self.layers = layers
        self.loss_function = loss_function
        # self.epochs = epochs
        # self.learning_rate = learning_rate

        self._check_valid_layers()

    def _check_valid_layers(self):
        for layer in self.layers:
            if not issubclass(layer, Layer):
                raise TypeError("layers argument should be a list of classes that inherit from Layer class")
        for fan_out, fan_in in zip(self.layers[:-1].dimensions[1], self.layers[1:].dimensions[0]):
            if fan_out != fan_in:
                raise ValueError("amount of output neurons of every layer should " + \
                                        "correspond to amount of input neurons in neXt layer")

    def fit(self, X, Y, epochs: int=1000, learning_rate: float=0.01, stochastic: bool=False):
        self._gradient_descent(X, Y, epochs, learning_rate, stochastic)

    def _gradient_descent(self, X, Y, epochs: int=1000, learning_rate: float=0.01, stochastic=False):
        error = 0
        for epoch in range(epochs):
            output = self._forward_propagation(X)

            error += self.loss_function.forward(Y, output)

            gradient = self.loss_function.backward(Y, output)
            self._backward_propagation(gradient, learning_rate)

    def _forward_propagation(self, X) -> np.array:
        output = X
        for layer in self.layers:
            output = layer.forward(output, activation=layer.activation)
        return output

    def _backward_propagation(self, gradient, learning_rate: float=0.01) -> np.array:
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient, learning_rate)
