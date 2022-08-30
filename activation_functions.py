from abc import ABC, abstractmethod
import numpy as np


class Activation(ABC):
    @staticmethod
    @abstractmethod
    def forward(z: np.array) -> np.array:
        pass

    @staticmethod
    @abstractmethod
    def backward(z: np.array, gradient: np.array) -> np.array:
        pass


class Sigmoid(Activation):
    @staticmethod
    def forward(z: np.array) -> np.array:
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def backward(z: np.array, gradient: np.array) -> np.array:
        s = Sigmoid.forward(z)
        return np.multiply(gradient, s * (1 - s))


class ReLU(Activation):
    @staticmethod
    def forward(z: np.array) -> np.array:
        return np.maximum(0, z)

    @staticmethod
    def backward(z: np.array, gradient: np.array) -> np.array:
        return np.multiply(z > 0, gradient)


class Tanh(Activation):
    @staticmethod
    def forward(z: np.array) -> np.array:
        return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

    @staticmethod
    def backward(z: np.array, gradient: np.array) -> np.array:
        return np.multiply(1 - Tanh.forward(z) ** 2, gradient)


class SoftMax(Activation):
    @staticmethod
    def forward(x: np.array, epsilon=0.000001) -> np.array:
        # z = np.exp(x) #numerically unstable
        z = np.exp(x - np.max(x))
        return z / np.sum(z)

    @staticmethod
    def backward2(x: np.array, gradient: np.array) -> np.array:
        s = x.reshape(-1, 1)
        gradient = gradient.reshape(-1, 1)
        return np.dot(np.diagflat(s) - np.dot(s, s.T), gradient)
        # s = x.reshape(-1, 1)
        # return np.dot(gradient, np.diagflat(s) - np.dot(s, s.T))

    @staticmethod
    def backward(x: np.array, gradient: np.array) -> np.array:
        # x = SoftMax.forward(x)
        x = x.reshape(-1, 1)
        gradient = gradient.reshape(-1, 1)
        n = np.size(x)
        return (gradient @ (np.identity(n) - x.T @ x))

    @staticmethod
    def backward3(x: np.array, gradient: np.array) -> np.array:
        # x = SoftMax.forward(x)
        x = x.reshape(1, -1)
        gradient = gradient.reshape(1, -1)
        n = np.size(x)
        return (gradient @ (np.identity(n) - x.T @ x)).T
    
    def backward4(x, gradient): # Best implementation (VERY FAST)
        s = SoftMax.forward(x)
        a = np.eye(s.shape[-1])
        temp1 = np.zeros((s.shape[0], s.shape[1], s.shape[1]),dtype=np.float32)
        temp2 = np.zeros((s.shape[0], s.shape[1], s.shape[1]),dtype=np.float32)
        temp1 = np.einsum('ij,jk->ijk',s,a)
        temp2 = np.einsum('ij,ik->ijk',s,s)
        print(temp1-temp2)
        return np.multiply(temp1-temp2, gradient)
