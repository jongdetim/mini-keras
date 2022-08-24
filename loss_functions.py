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
        print("truth is: ", truth)
        size = np.shape(truth)[0] #or [1] ?
        print("size is ", size)
        return -1 / size * np.sum(((truth * np.log(prediction)) + ((1 - truth) * np.log(1 - prediction))))

    @staticmethod
    def backward(prediction: np.array, truth: np.array) -> np.array:
        return ((1 - truth) / (1 - prediction) - truth / prediction) / np.size(truth)
