from abc import ABC, abstractmethod
from typing import Callable
import numpy as np


class Layer(ABC):
    def __init__(self, dimensions: tuple, activation: Callable=None):
        self.input = None
        self.output = None
        self.activation = activation
        self.dimensions = dimensions

    @abstractmethod
    def forward(self, x, activation=None):
        pass

    @abstractmethod
    def backward(self, output_gradient, learning_rate, activation=None):
        pass


class Dense(Layer):
    allowed_distributions = ['uniform', 'normal', 'gaussian']
    allowed_init_methods = ['xavier', 'normal', ]

    def __init__(self, dimensions: tuple, activation: Callable=None, weights=None,
                 biases=None, init_method='xavier', distribution='uniform'):
        self.weights: np.array = np.array(weights) if weights is not None else None
        self.biases: np.array = np.array(biases) if biases is not None else None
        self.init_method = init_method
        self.distribution = distribution
        super().__init__(dimensions, activation)

        self._check_valid_args()

        if self.weights is None:
            self._init_random_params(self.dimensions)

    def _check_valid_args(self):
        if len(self.dimensions) != 2 or any in self.dimensions <= 0:
            raise ValueError("dimensions has to be a tuple of 2 positive integers")
        if self.weights is not None and self.weights.shape != self.dimensions:
            raise ValueError(
                f"weight matrix shape: {self.weights.shape} does not correspond to layer dimensions: {self.dimensions}")
        if self.init_method not in self.allowed_init_methods:
            raise ValueError(f"init method: '{self.init_method}' is not valid")
        if self.distribution not in self.allowed_distributions:
            raise ValueError(f"distribution: '{self.distribution}' is not valid")

    def _init_random_params(self, dimensions: tuple):
        fan_in, fan_out = dimensions
        if self.init_method == 'xavier':
            self.weights = self._xavier_init(fan_in, fan_out)
        elif self.init_method == 'normal':
            self.weights = self._normal_init(fan_in, fan_out)
        self.biases = np.zeros(fan_out)

    def _normal_init(self, fan_in: int, fan_out: int) -> np.array:
        if self.distribution in ['normal','gaussian']:
            return np.random.normal(loc=0.0, scale=0.02, size=(fan_in, fan_out))
        if self.distribution == 'uniform':
            return np.random.uniform(low=-0.05, high=0.05, size=(fan_in, fan_out))
        raise ValueError("distribution is invalid!")

    def _xavier_init(self, fan_in: int, fan_out: int) -> np.array:
        if self.distribution in ['normal','gaussian']:
            limit = np.sqrt(2 / float(fan_in + fan_out))
            return np.random.normal(loc=0.0, scale=limit, size=(fan_in, fan_out))
        if self.distribution == 'uniform':
            limit = np.sqrt(6 / float(fan_in + fan_out))
            return np.random.uniform(low=-limit, high=limit, size=(fan_in, fan_out))
        raise ValueError("distribution is invalid!")

    def forward(self, x, activation=None) -> np.array:
        self.input = x
        z = self.weights.T.dot(x) + self.biases
        self.output = z #<- currently saves output BEFORE activation function is applied!! not sure if correct
        if activation:
            z = activation.forward(z)
        return z

    def backward(self, output_gradient, learning_rate, activation=None):
        gradient = np.multiply(output_gradient, activation.backward(self.output)) #<- gotta check if this is correct!
        weights_gradient = np.dot(gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, gradient)
        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * gradient
        return input_gradient
