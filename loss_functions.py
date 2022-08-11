from abc import ABC, abstractmethod
import numpy as np


class Loss(ABC):
    @staticmethod
    @abstractmethod
    def forward(x: np.array) -> np.array:
        pass

    @staticmethod
    @abstractmethod
    def backward(x: np.array) -> np.array:
        pass


class BinaryCrossEntropy(Loss):
    @staticmethod
    def forward(x: np.array) -> np.array:
        pass

    @staticmethod
    def backward(x: np.array) -> np.array:
        pass

