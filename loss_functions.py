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
    def forward(prediction: np.array, truth: np.array) -> np.array:
        size = len(truth)
#         return -1 / size * ((truth * np.log(prediction)) + ((1 - truth) * np.log(1 - prediction)))
        return -1 / size * np.sum(((truth * np.log(prediction)) + ((1 - truth) * np.log(1 - prediction))))

    @staticmethod
    def backward(prediction: np.array, truth: np.array) -> np.array:
        return -(truth / prediction) + ((1 - truth) / (1 - prediction))
