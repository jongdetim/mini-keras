from abc import ABC, abstractmethod
from typing import Callable
import numpy as np


class BaseLayer(ABC):
    """Abstract Base Class / blueprint for layers used by the Sequential model"""
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


class Dense(BaseLayer):
    """Fully connected layer class..

    Arguments
    ---------
        dimensions (tuple):
            Tuple of (n input neurons, m output neurons)
        activation (Callable, optional):
            Activation function to be applied on output. Defaults to None.
        weights (numpy.ndarray, optional):
            Weights to initialize Dense instance with. Weight dimensions have to align with the specified dimensions parameter. Defaults to None.
        biases (numpy.ndarray, optional):
            Biases to initialize Dense instance with. Defaults to None.
        init_method (str, optional):
            Weight initialization method. Valid values are ['xavier', 'normal']. Defaults to 'xavier'.
        distribution (str, optional):
            Weight initialiazation distribution. Valid values are ['uniform', 'normal' / 'gaussian']. Defaults to 'uniform'.
        seed (int, optional):
            Sets seed for random weight initialization. Defaults to None.

    Returns
    -------
        Dense object: Class instance with initialized weights & biases.
    """
    allowed_init_methods = ['xavier', 'normal']
    allowed_distributions = ['uniform', 'normal', 'gaussian']

    def __init__(self, dimensions: tuple, activation: Callable=None, weights: np.ndarray=None, biases: np.ndarray=None, init_method='xavier', distribution='uniform', seed: int=None):
        self.weights: np.array = np.array(weights) if weights is not None else None
        self.biases: np.array = np.array(biases) if biases is not None else None
        self.init_method = init_method
        self.distribution = distribution
        super().__init__(dimensions, activation)

        self._check_valid_args()

        if self.weights is None:
            self._init_random_params(self.dimensions, seed)

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

    def _init_random_params(self, dimensions: tuple, seed: int):
        if seed is not None:
            np.random.seed(seed)
        fan_in, fan_out = dimensions
        if self.init_method == 'xavier':
            self.weights = self._xavier_init(fan_in, fan_out)
        elif self.init_method == 'normal':
            self.weights = self._normal_init(fan_in, fan_out)
        self.biases = np.zeros((fan_out, 1))

    def _normal_init(self, fan_in: int, fan_out: int) -> np.ndarray:
        if self.distribution in ['normal','gaussian']:
            return np.random.normal(loc=0.0, scale=0.02, size=(fan_out, fan_in))
        if self.distribution == 'uniform':
            return np.random.uniform(low=-0.05, high=0.05, size=(fan_out, fan_in))
        raise ValueError("distribution is invalid!")

    def _xavier_init(self, fan_in: int, fan_out: int) -> np.ndarray:
        if self.distribution in ['normal','gaussian']:
            limit = np.sqrt(2 / float(fan_in + fan_out))
            return np.random.normal(loc=0.0, scale=limit, size=(fan_out, fan_in))
        if self.distribution == 'uniform':
            limit = np.sqrt(6 / float(fan_in + fan_out))
            return np.random.uniform(low=-limit, high=limit, size=(fan_out, fan_in))
        raise ValueError("distribution is invalid!")

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the layer. Takes the dotproduct of X and weights, and adds bias.

        Arguments
        ---------
            X (numpy.ndarray):
                Layer input.

        Returns
        -------
            numpy.ndarray: Layer output.
        """
        self.input = x
        z = self.weights.dot(x) + self.biases
        self.output = z
        if self.activation:
            z = self.activation.forward(z)
        return z

    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        """Backward pass through the layer. Takes the output gradient and adjusts weights and biases.

        Arguments
        ---------
            output_gradient (numpy.ndarray):
                Gradient from the next layer.
            learning_rate (float):
                Learning rate to multiply the parameter adjustments based on the gradient with.

        Returns
        -------
            numpy.ndarray:
                Gradient of the layer input.
        """
        gradient = self.activation.backward(self.output, output_gradient)
        weights_gradient = np.dot(gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, gradient)
        self.weights -= learning_rate * weights_gradient / self.input.shape[1]
        self.biases -= learning_rate * np.mean(gradient, axis=1, keepdims=True)
        return input_gradient
