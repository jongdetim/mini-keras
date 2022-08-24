from abc import ABC, abstractmethod
import numpy as np


class Activation(ABC):
    @staticmethod
    @abstractmethod
    def forward(x: np.array) -> np.array:
        pass

    @staticmethod
    @abstractmethod
    def backward(x: np.array, gradient: np.array) -> np.array:
        pass


class Sigmoid(Activation):
    @staticmethod
    def forward(x: np.array) -> np.array:
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def backward(x: np.array, gradient: np.array) -> np.array:
        s = Sigmoid.forward(x)
        return np.multiply(s * (1 - s), gradient)


class ReLU(Activation):
    @staticmethod
    def forward(x: np.array) -> np.array:
        return np.maximum(0, x)

    @staticmethod
    def backward(x: np.array, gradient: np.array) -> np.array:
        return np.multiply(x > 0, gradient)


class Tanh(Activation):
    @staticmethod
    def forward(x: np.array) -> np.array:
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    @staticmethod
    def backward(x: np.array, gradient: np.array) -> np.array:
        return np.multiply(1 - Tanh.forward(x) ** 2, gradient)


class SoftMax(Activation):
    @staticmethod
    def forward(x: np.array) -> np.array:
        z = np.exp(x)
        # z = np.exp(x - np.max(x))
        return z / np.sum(z)

    # @staticmethod
    # def backward(x: np.array) -> np.array:
    #     s = x.reshape(-1, 1)
    #     return np.diagflat(s) - np.dot(s, s.T)

    @staticmethod
    def backward(x: np.array, gradient: np.array) -> np.array:
        n = np.size(x)
        return np.dot((np.identity(n) - x.T) * x, gradient)
