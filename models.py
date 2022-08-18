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

    def fit(self, X, Y, epochs: int = 1000, learning_rate: float = 0.01, stochastic: bool = False):
        self._gradient_descent(X, Y, epochs, learning_rate, stochastic)

    def _gradient_descent(self, X, Y, epochs: int = 1000, learning_rate: float = 0.01, stochastic=False):
        error = 0
        for epoch in range(epochs):
            output = self._forward_propagation(X)

            print(output)
            error += self.loss_function.forward(output, Y)
            # gebruiken we de error niet voor loss function backprop?
            gradient = self.loss_function.backward(output, Y)
            self._backward_propagation(gradient, learning_rate)

    def _forward_propagation(self, X) -> np.array:
        output = X
        for layer in self.layers:
            output = layer.forward(output, activation=layer.activation)
        return output

    def _backward_propagation(self, gradient, learning_rate: float = 0.01) -> np.array:
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient, learning_rate)
