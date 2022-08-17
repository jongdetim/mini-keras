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
        pass

    @staticmethod
    def backward(prediction: np.array, truth: np.array) -> np.array:
        pass

