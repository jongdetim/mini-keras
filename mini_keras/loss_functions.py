from abc import ABC, abstractmethod
import numpy as np


class BaseLoss(ABC):
    @staticmethod
    @abstractmethod
    def forward(prediction: np.array, truth: np.array) -> np.array:
        pass

    @staticmethod
    @abstractmethod
    def backward(prediction: np.array, truth: np.array) -> np.array:
        pass


class BinaryCrossEntropy(BaseLoss):
    @staticmethod
    def forward(prediction: np.array, truth: np.array, epsilon=0.000001) -> np.array:
        prediction = np.clip(prediction, epsilon, 1 - epsilon)
        size = truth.shape[1]
        return -1 / size * np.sum(((truth * np.log(prediction)) + ((1 - truth) * np.log(1 - prediction))))

    @staticmethod
    def backward(prediction: np.array, truth: np.array, epsilon=0.0000001) -> np.array:
        prediction = np.clip(prediction, epsilon, 1 - epsilon)
        return (prediction - truth) / (prediction * (1 - prediction)) / truth.shape[0]
