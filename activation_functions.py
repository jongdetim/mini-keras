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
        return np.multiply(s * (1 - s), gradient)


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
        return z / np.sum(z, axis=0)

    # @staticmethod
    # def backward(x: np.array, gradient: np.array) -> np.array:
    #     # s = SoftMax.forward(x).reshape(-1, 1)
    #     s = SoftMax.forward(x)
    #     print(s)
    #     print(s.reshape(-1, 1))
    #     # print("deriv:", np.diagflat(s) - np.dot(s, s.T))
    #     # print(x)
    #     # return np.dot(np.diagflat(s) - np.dot(s, s.T), gradient)
    #     # result = np.einsum('ijk,ki->ji', np.diagflat(s) - np.dot(s, s.T), gradient)
    #     return np.diagflat(s) - np.dot(s, s.T)

    # @staticmethod
    # def backward(x: np.array, gradient: np.array) -> np.array:
    #     x.reshape(-1, 1)
    #     I = np.eye(x.shape[0])
    #     return np.dot(SoftMax.forward(x) * (I - SoftMax.forward(x).T), gradient)

    # @staticmethod
    # def backward5(x: np.array, gradient: np.array) -> np.array:
    #     x = SoftMax.forward(x)
    #     x = x.reshape(-1, 1)
    #     gradient = gradient.reshape(1, -1)
    #     n = np.size(x)
    #     # print(np.identity(n) - x.T @ x)
    #     # print(gradient)
    #     return np.dot(gradient, np.identity(n) - x.T @ x).T

    # @staticmethod
    # def backward3(x: np.array, gradient: np.array) -> np.array:
    #     x = SoftMax.forward(x)
    #     # x = x.reshape(-1, 1)
    #     # gradient = gradient.reshape(1, -1)
    #     n = np.size(x)
    #     return np.identity(n) - x.T @ x

    @staticmethod
    def backward(x, gradient):  # Best implementation (VERY FAST)
        s = SoftMax.forward(x).T
        a = np.eye(s.shape[-1])
        # temp1 = np.zeros((s.shape[0], s.shape[1], s.shape[1]),dtype=np.float32)
        # temp2 = np.zeros((s.shape[0], s.shape[1], s.shape[1]),dtype=np.float32)
        temp1 = np.einsum('ij,jk->ijk', s, a)
        temp2 = np.einsum('ij,ik->ijk', s, s)
        # print("softmax_derivative:", temp1-temp2)
        result = np.einsum('ijk,ki->ji', temp1 - temp2, gradient)
        # print("softmax:", s)
        # print("gradient:", gradient)
        # print("result:", result)
        return result
        # return np.dot(temp1-temp2, gradient)

    # @staticmethod
    # def backward7(x, gradient):  # Best implementation (VERY FAST)
    #     s = SoftMax.forward(x)
    #     a = np.eye(s.shape[-1])
    #     # temp1 = np.zeros((s.shape[0], s.shape[1], s.shape[1]),dtype=np.float32)
    #     # temp2 = np.zeros((s.shape[0], s.shape[1], s.shape[1]),dtype=np.float32)
    #     temp1 = np.einsum('ij,kj->ijk', s, a)
    #     temp2 = np.einsum('ij,ik->ijk', s, s)
    #     # print(temp1, temp2, temp1-temp2)
    #     deriv = np.array([[[0.20110808, -0.20110808], [-0.20110808, 0.20110808]],
    #                       [[0.3, -0.2], [-0.5, 0.8]]])
    #     # result = np.einsum('ijk,jk->jk', temp1-temp2, gradient)
    #     print("deriv shape:", deriv.shape)
    #     result = np.einsum('ijk,ki->ji', deriv, gradient)
    #     print("deriv:", deriv)
    #     print("softmax:", s)
    #     print("softmax_derivative:", temp1-temp2)
    #     print("gradient:", gradient)
    #     print("result:", result)
    #     return result
    #     # return np.dot(temp1-temp2, gradient)

    # @staticmethod
    # def backward6(x, gradient):
    #     result = np.array([SoftMax.backward(row, gradient) for row in x.T])
    #     return np.einsum('ijk,ki->ji', result, gradient)
