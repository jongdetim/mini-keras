from abc import ABC, abstractmethod
import numpy as np


class Loss(ABC):
    @staticmethod
    @abstractmethod
    def forward(prediction: np.array, truth: np.array) -> np.array:
        pass

    @staticmethod
    @abstractmethod
    def backward(prediction: np.array, truth: np.array) -> np.array:
        pass


class BinaryCrossEntropy(Loss):
    @staticmethod
    def forward(prediction: np.array, truth: np.array, epsilon=0.000001) -> np.array:
        prediction = np.clip(prediction, epsilon, 1 - epsilon)
        size = np.shape(truth)[0] #or [1] ?
        return -1 / size * np.sum(((truth * np.log(prediction)) + ((1 - truth) * np.log(1 - prediction))))

    @staticmethod
    def backward(prediction: np.array, truth: np.array, epsilon=0.000001) -> np.array:
        # return np.array([-0.5, 0.5]).reshape(-1, 1)
            
        prediction = np.clip(prediction, epsilon, 1 - epsilon)
        return (prediction - truth) / (prediction * (1 - prediction))
        # prediction = np.clip(prediction, epsilon, 1 - epsilon)
        # return (prediction - truth) / (prediction * (1 - prediction)) / np.size(truth)
