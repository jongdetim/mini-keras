import numpy as np


class DenseNeuralNet:
    """Fully connected artificial neural network model

    Parameters:
        dimensions (list / NumPy array): Takes a 1-dimensional vector of length n integers,
        where each element specifies the amount of neurons in that layer of the model, including
        input and output layers. The length of the array determines the total amount of layers.

        weights (list / NumPy array, optional): Weights to initialize the model.
        Weight dimensions have to align with the specified dimensions parameter. Defaults to None.

        init_method (str, optional): weight initialization method.
        Valid values are ['xavier', 'normal']. Defaults to 'xavier'.

        distribution (str, optional): Weight initialiazation distribution.
        Valid values are ['uniform', 'normal' / 'gaussian']. Defaults to 'uniform'.

    Returns:
        DenseNeuralNet object: Class instance with initialized weights & biases.
    """
    allowed_distributions = ['uniform', 'normal', 'gaussian']
    allowed_init_methods = ['xavier', 'normal', ]

    def __init__(self, dimensions: np.array | list, weights: np.array | list = None, init_method='xavier', distribution='uniform'):
        self.weights: list = weights if weights is not None else []
        self.biases: list = []
        self.init_method = init_method
        self.distribution = distribution

        if self.init_method not in self.allowed_init_methods:
            raise ValueError("init method: '" +
                             self.init_method + "' is not valid")
        if self.distribution not in self.allowed_distributions:
            raise ValueError("distribution: '" +
                             self.distribution + "' is not valid")
        if not self.weights:
            self._init_random_params(dimensions)
        print(self.weights, self.biases)

    def _init_random_params(self, dimensions):
        for fan_in, fan_out in zip(dimensions[:-1], dimensions[1:]):
            if self.init_method == 'xavier':
                self.weights.append(self._xavier_init(fan_in, fan_out))
            elif self.init_method == 'normal':
                self.weights.append(self._normal_init(fan_in, fan_out))
            self.biases.append(np.zeros(fan_out))

    def _normal_init(self, fan_in: int, fan_out: int) -> np.array:
        if self.distribution == ('normal' or 'gaussian'):
            return np.random.normal(loc=0.0, scale=1.0, size=(fan_in, fan_out)) * 0.01
        if self.distribution == 'uniform':
            return np.random.uniform(low=-0.05, high=0.05, size=(fan_in, fan_out))
        raise ValueError("distribution is invalid!")

    def _xavier_init(self, fan_in: int, fan_out: int) -> np.array:
        if self.distribution == ('normal' or 'gaussian'):
            limit = np.sqrt(2 / float(fan_in + fan_out))
            return np.random.normal(loc=0.0, scale=limit, size=(fan_in, fan_out))
        if self.distribution == 'uniform':
            limit = np.sqrt(6 / float(fan_in + fan_out))
            return np.random.uniform(low=-limit, high=limit, size=(fan_in, fan_out))
        raise ValueError("distribution is invalid!")

    def _forward_propagation(self):
        pass

    def _sigmoid(self, x: np.array) -> np.array:
        return 1 / (1 + np.exp(-x))

    def _relu(self, x: np.array) -> np.array:
        return np.maximum(0, x)

    def _softmax(self, x) -> np.array:
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)
