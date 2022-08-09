from abc import ABC, abstractmethod
import numpy as np


class Activation(ABC):
    @abstractmethod
    def forward(self, x: np.array) -> np.array:
        pass

    @abstractmethod
    def backward(self, x: np.array) -> np.array:
        pass


class Sigmoid(Activation):
    def forward(self, x: np.array) -> np.array:
        return 1 / (1 + np.exp(-x))

    def backward(self, x: np.array) -> np.array:
        forward = self.forward(x)
        return forward * (1 - forward)


class ReLU(Activation):
    def forward(self, x: np.array) -> np.array:
        return np.maximum(0, x)

    def backward(self, x: np.array) -> np.array:
        return x > 0


class Tanh(Activation):
    def forward(self, x: np.array) -> np.array:
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def backward(self, x: np.array) -> np.array:
        return 1 - self.forward(x) ** 2


class SoftMax(Activation):
    def forward(self, x: np.array) -> np.array:
        z = np.exp(x)
        return z / np.sum(z)

    def backward(self, x: np.array) -> np.array:
        s = x.reshape(-1, 1)
        return np.diagflat(s) - np.dot(s, s.T)
