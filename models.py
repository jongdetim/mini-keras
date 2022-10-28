from typing import List, Callable
from math import ceil
import pickle
import numpy as np
import matplotlib.pyplot as plt

from layers import Layer
# from loss_functions import *
from utils import shuffle_arrays, split_given_size


class Sequential:
    """Fully connected artificial neural network model

    Arguments
    ---------
        layers (list(Layer)):
            Takes a list of class instances that inherit from Layer baseclass.
            The amount of output neurons of every layer should correspond to the amount of input neurons in the next layer.

        loss_function (Callable):
            Takes callable object to evaluate the loss on set of data.

    Returns
    -------
        Sequential object: Class instance with initialized weights & biases.

    Example
    -------
        model = Sequential([Dense((30, 8), activation=Sigmoid),
                            Dense((8, 5), activation=LReLU),
                            Dense((5, 2), activation=SoftMax)], BinaryCrossEntropy)
    """
    allowed_output_types = ['numerical', 'exclusive', 'multi-label']

    def __init__(self, layers: List[Layer], loss_function: Callable):
        self.layers = layers
        self.loss_function = loss_function

        self._check_valid_layers()

    def _check_valid_layers(self):
        if not self.layers:
            raise ValueError(
                "layers argument is empty. should contain atleast one layer")
        for layer in self.layers:
            if not isinstance(layer, Layer):
                raise TypeError(
                    "layers argument should be a list of classes that inherit from Layer class")
        if len(self.layers) > 1:
            for fan_out, fan_in in zip(self.layers[:-1], self.layers[1:]):
                if fan_out.dimensions[1] != fan_in.dimensions[0]:
                    raise ValueError("amount of output neurons of every layer should " +
                                     "correspond to amount of input neurons in next layer")

    def _check_valid_data(self, X, Y):
        if len(X) != len(Y):
            raise ValueError("X must contain the same amount of samples as Y")

    def fit(self, X, Y, epochs: int = 1000, learning_rate: float = 0.01, batch_size: int = 32, verbose: bool = True, seed: int = None, validation_set=None) -> None:
        """Trains the model by using (batch/stochastic) gradient descent

        Arguments
        ---------
            X (numpy.ndarray):
                Input data to train the model on. Features should be on 1st dimension, rows on 2nd dimension. Rows should correspond to Y rows.
            Y (numpy.ndarray):
                Output labels / truth. If categorical, this has to be a one-hot encoded matrix. Rows should correspond to X rows.
            epochs (int, optional):
                Amount of iterations to go over the entire dataset. Defaults to 1000.
            learning_rate (float, optional):
                Scalar to be multiplied with each learning step / parameter update. Defaults to 0.01.
            batch_size (int, optional):
                Size of batches to feed to the forward pass. Learning step / parameter update is done after every batch.
                If the final batch is smaller than batch_size, the remaining data is forward-fed, and the learning step / parameters update is scaled to the relative size of the batch.
                Defaults to 32.
            verbose (bool, optional):
                Flag to print learning process on stdout. Defaults to True.
            seed (int, optional):
                Seed for data shuffling at the start of every epoch. Defaults to None.
            validation_set (dict(['X', 'Y']), optional):
                Dictionary with 'X' key mapped to X feature datapoints value and 'Y' key mapped to Y output datapoints. Defaults to None.
        """
        self._check_valid_data(X, Y)
        if not Y.shape[1] == self.layers[-1].dimensions[1]:
            raise ValueError(
                f"Y size ({Y.shape[1]}) should be equal to output layer size ({self.layers[-1].dimensions[1]})")
        if batch_size > len(X) or batch_size < 1:
            raise ValueError(
                f"batch_size ({batch_size}) should be integer value between 1 and amount of datapoints ({len(X)})")
        self._batch_gradient_descent(
            X, Y, epochs, learning_rate, batch_size, verbose, seed, validation_set)

    # could be part of base class Model, that Sequential would inherit from
    def predict(self, X: np.ndarray, Y_labels: np.ndarray = None, output_type: str = 'numerical', cutoff: float = 0.5) -> np.ndarray:
        """Predicts output based on X input data.

        Arguments
        ---------
            X (numpy.ndarray):
                Input data. Features should be on 1st dimension, rows on 2nd dimension.
            Y_labels (numpy.ndarray, optional):
                labels corresponding to output indices, only for when output data is categorical. Defaults to None.
            output_type (str, optional):
                type of output data. Valid values are ['numerical', 'exclusive', 'multi-label']. Defaults to 'numerical'.
            cutoff (float, optional):
                When output type is 'multi-label', this sets the cutoff value above which a label is predicted or not. Defaults to 0.5.

        Returns
        -------
            numpy.ndarray: Matrix with predicted output values.
        """
        if output_type not in self.allowed_output_types:
            raise ValueError(
                f"output_type has to be one of {self.allowed_output_types}")
        if Y_labels is not None and len(Y_labels) is not self.layers[-1].dimensions[1]:
            raise ValueError(
                "Y_labels has to be same length as output neurons")

        X = X.reshape(-1, X.shape[0]).T if len(X.shape) < 2 else X.T
        Y = self._forward_propagation(X)

        if output_type == 'numerical':
            return Y.T
        if output_type == 'exclusive':
            return np.argmax(Y, axis=0) if Y_labels is None else Y_labels[np.argmax(Y, axis=0)]
        if output_type == 'multi-label':
            return Y.T >= cutoff if Y_labels is None \
                else np.array(list((Y_labels[(Y >= cutoff)[:, i]] for i in range(Y.shape[1]))), dtype=object)

    def score(self, X: np.ndarray, Y: np.ndarray) -> np.float32:
        """Calculates the accuracy of the model on a given dataset

        Arguments
        ---------
            X (numpy.ndarray):
                Input data. Features should be on 1st dimension, rows on 2nd dimension.
            Y (numpy.ndarray):
                Output labels / truth. If categorical, this has to be a one-hot encoded matrix. Rows should correspond to X rows.

        Returns
        -------
            numpy.float32: Percentage of correct predictions.
        """
        prediction = self.predict(X, output_type='exclusive')
        score = np.sum(prediction == np.argmax(Y, axis=1)) / len(prediction)
        return score

    def loss(self, X: np.ndarray, Y: np.ndarray) -> np.float32:
        """Does a forward pass and calculates loss

        Arguments
        ---------
            X (np.ndarray):
                Input data. Features should be on 1st dimension, rows on 2nd dimension.
            Y (np.ndarray):
                Output labels / truth. If categorical, this has to be a one-hot encoded matrix. Rows should correspond to X rows.

        Returns
        -------
            numpy.float32: loss function output value over provided data
        """
        X = X.T
        Y = Y.reshape(Y.shape[0], -1).T
        pred = self._forward_propagation(X)
        return self.loss_function.forward(pred, Y)

    def _prepare_batches(self, X, Y, batch_size, seed):
        X, Y = shuffle_arrays([X, Y], seed)
        return split_given_size(X, batch_size), split_given_size(Y, batch_size)

    def _batch_gradient_descent(self, X, Y, epochs, learning_rate, batch_size, verbose, seed, plot=True, validation_set=None):
        errors = []
        validation_errors = []
        for epoch in range(epochs):
            # epoch_error = 0
            x_batches, y_batches = self._prepare_batches(
                X, Y, batch_size, seed) if batch_size < len(X) else ([X], [Y])
            for x_batch, y_batch in zip(x_batches, y_batches):
                output = self._forward_propagation(x_batch.T)
                # epoch_error += self.loss_function.forward(output, y_batch.T)
                gradient = self.loss_function.backward(output, y_batch.T)
                self._backward_propagation(
                    gradient, learning_rate * x_batch.shape[0] / batch_size)

            # errors.append(epoch_error / len(x_batches))
            if verbose or plot:
                errors.append(self.loss(X, Y))

            if validation_set is not None:
                validation_errors.append(
                    self.loss(validation_set['X'], validation_set['Y']))

            if verbose and (epoch + 1) % (ceil(epochs / 10) if epochs >= 10 else 2) == 0:
                if validation_set is None:
                    print(
                        f"epoch: {epoch + 1}/{epochs} - training loss: {errors[-1]}")
                else:
                    print(
                        f"epoch: {epoch + 1}/{epochs} - training loss: {errors[-1]} - validation loss: {self.loss(validation_set['X'], validation_set['Y'])}")

        if plot:
            self._plot_error(errors, validation_errors)

    def _forward_propagation(self, X) -> np.array:
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def _backward_propagation(self, gradient, learning_rate) -> np.array:
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient, learning_rate)

    def _plot_error(self, errors, validation_errors=None):
        plt.plot(range(0, len(errors)), errors, 'r', label='training set')
        if validation_errors is not None:
            plt.plot(range(0, len(validation_errors)),
                     validation_errors, 'b', label='validation set')
        plt.locator_params(axis="x", integer=True, tight=True)
        plt.ylabel("loss")
        plt.xlabel("epochs")
        plt.legend()
        plt.show()

    def save(self, file_path: str) -> None:
        """Saves the Sequential class instance to file as binary .pickle file.

        Arguments
        ---------
            file_path (str):
                path and filename to write the model to. adds .pickle extension.
        """
        with open(file_path + '.pickle', 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(file_path: str) -> object:
        """Loads the Sequential class instance from binary .pickle file.

        Arguments
        ---------
            file_path (str):
                path to file to read the model from.

        Returns
        -------
            Sequential object: Class instance loaded from file.
        """
        with open(file_path, 'rb') as file:
            return pickle.load(file)
