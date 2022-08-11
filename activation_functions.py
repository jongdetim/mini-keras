from abc import ABC, abstractmethod
import numpy as np


class Activation(ABC):
    @staticmethod
    @abstractmethod
    def forward(x: np.array) -> np.array:
        pass

    @staticmethod
    @abstractmethod
    def backward(x: np.array) -> np.array:
        pass


class Sigmoid(Activation):
    @staticmethod
    def forward(x: np.array) -> np.array:
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def backward(x: np.array) -> np.array:
        s = Sigmoid.forward(x)
        return s * (1 - s)


class ReLU(Activation):
    @staticmethod
    def forward(x: np.array) -> np.array:
        return np.maximum(0, x)

    @staticmethod
    def backward(x: np.array) -> np.array:
        return x > 0


class Tanh(Activation):
    @staticmethod
    def forward(x: np.array) -> np.array:
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    @staticmethod
    def backward(x: np.array) -> np.array:
        return 1 - Tanh.forward(x) ** 2


class SoftMax(Activation):
    @staticmethod
    def forward(x: np.array) -> np.array:
        z = np.exp(x)
        return z / np.sum(z)

    @staticmethod
    def backward(x: np.array) -> np.array:
        s = x.reshape(-1, 1)
        return np.diagflat(s) - np.dot(s, s.T)
