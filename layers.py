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
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, output_gradient, learning_rate):
        pass


class Dense(Layer):
    allowed_distributions = ['uniform', 'normal', 'gaussian']
    allowed_init_methods = ['xavier', 'normal']

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
        self.biases = np.zeros((fan_out, 1))

    def _normal_init(self, fan_in: int, fan_out: int) -> np.array:
        if self.distribution in ['normal','gaussian']:
            return np.random.normal(loc=0.0, scale=0.02, size=(fan_out, fan_in))
        if self.distribution == 'uniform':
            return np.random.uniform(low=-0.05, high=0.05, size=(fan_out, fan_in))
        raise ValueError("distribution is invalid!")

    def _xavier_init(self, fan_in: int, fan_out: int) -> np.array:
        if self.distribution in ['normal','gaussian']:
            limit = np.sqrt(2 / float(fan_in + fan_out))
            return np.random.normal(loc=0.0, scale=limit, size=(fan_out, fan_in))
        if self.distribution == 'uniform':
            limit = np.sqrt(6 / float(fan_in + fan_out))
            return np.random.uniform(low=-limit, high=limit, size=(fan_out, fan_in))
        raise ValueError("distribution is invalid!")

    def forward(self, x) -> np.array:
        self.input = x
        # print(x.shape)
        # print(self.biases.shape)
        # print(self.weights.T.dot(x))
        z = self.weights.dot(x) + self.biases
        self.output = z
        if self.activation:
            z = self.activation.forward(z)
        return z

    # def backward(self, output_gradient, learning_rate):
    #     gradient = np.multiply(output_gradient, self.activation.backward(self.output)) #<- gotta check if this is correct!
    #     weights_gradient = np.dot(gradient, self.input.T)
    #     input_gradient = np.dot(self.weights.T, gradient)
    #     self.weights -= learning_rate * weights_gradient
    #     self.biases -= learning_rate * gradient
    #     return input_gradient

    def backward(self, output_gradient, learning_rate):
        gradient = self.activation.backward(self.output, output_gradient) #<- gotta check if this is correct!
        # print(output_gradient, self.output, gradient)
        # print("activation gradient:", gradient)
        # print("self.input.T:", self.input.T)
        weights_gradient = np.dot(gradient, self.input.T)
        # print("weights gradient:", weights_gradient)
        # print("weights:", self.weights)
        input_gradient = np.dot(self.weights.T, gradient)
        # print("input_gradient:", input_gradient)
        # print("shapes:", self.weights.shape, self.biases.shape)
        self.weights -= learning_rate * weights_gradient / self.input.shape[1]
        # print(np.sum(gradient, axis=1))
        # print(self.input.shape[1])
        # print(np.mean(gradient, axis=1))
        # print(self.biases)
        # print(self.biases.shape)
        # print(np.mean(gradient, axis=1).T.shape)
        self.biases -= learning_rate * np.mean(gradient, axis=1, keepdims=True)
        return input_gradient

    # def backward(self, output_gradient, learning_rate):
    #     print(output_gradient, self.output, self.activation.backward(self.output))
    #     gradient = np.dot(output_gradient, self.activation.backward(self.output)).reshape(-1,1) #<- gotta check if this is correct!
    #     print("gradient:", gradient, "self.input:", self.input, "self.weights:", self.weights)
    #     weights_gradient = np.dot(gradient, self.input.reshape(1, -1))
    #     input_gradient = np.dot(self.weights, gradient)
    #     print(weights_gradient.shape, self.weights.shape)
    #     self.weights -= learning_rate * weights_gradient.T
    #     self.biases -= learning_rate * gradient
    #     print(weights_gradient, "dad", input_gradient)
    #     return input_gradient
