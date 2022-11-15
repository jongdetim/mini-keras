from abc import ABC, abstractmethod
import numpy as np


class Activation(ABC):
    @staticmethod
    @abstractmethod
    def forward(z: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    @abstractmethod
    def backward(z: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        pass


class Sigmoid(Activation):
    @staticmethod
    def forward(z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def backward(z: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        s = Sigmoid.forward(z)
        return np.multiply(s * (1 - s), gradient)


class ReLU(Activation):
    @staticmethod
    def forward(z: np.ndarray) -> np.ndarray:
        return np.maximum(0, z)

    @staticmethod
    def backward(z: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        return np.multiply(z > 0, gradient)


class LReLU(Activation):
    @staticmethod
    def forward(z: np.ndarray) -> np.ndarray:
        return np.where(z > 0, z, z * 0.01)

    @staticmethod
    def backward(z: np.ndarray, gradient: np.ndarray, alpha=0.01) -> np.ndarray:
        return np.multiply(np.where(z > 0, 1, alpha), gradient)


class Tanh(Activation):
    @staticmethod
    def forward(z: np.ndarray) -> np.ndarray:
        return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

    @staticmethod
    def backward(z: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        return np.multiply(1 - Tanh.forward(z) ** 2, gradient)


class SoftMax(Activation):
    @staticmethod
    def forward(z: np.ndarray) -> np.ndarray:
        # z = np.exp(x) #numerically unstable
        x = np.exp(z - np.max(z))
        return x / np.sum(x, axis=0)

    @staticmethod
    def backward(z: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        s = SoftMax.forward(z).T
        a = np.eye(s.shape[-1])
        temp1 = np.einsum('ij,jk->ijk', s, a)
        temp2 = np.einsum('ij,ik->ijk', s, s)
        result = np.einsum('ijk,ki->ji', temp1 - temp2, gradient)
        return result
