from typing import Tuple
import numpy as np
import pandas as pd

def shuffle_in_unison(arrays, seed=None):
    """identical performance to shuffle_arrays"""
    n_elem = arrays[0].shape[0]
    if seed is not None:
        np.random.seed(seed)
    indices = np.random.permutation(n_elem)
    return (array[indices] for array in arrays)

def shuffle_arrays(arrays, seed=None):
    """Shuffles copies of arrays in unison, along axis=0

    Parameters:
    -----------
    arrays : List of NumPy arrays.
    seed : Seed value if int >= 0, else seed is random.
    """
    perm = np.arange(arrays[0].shape[0])
    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(perm)
    return (array[perm] for array in arrays)

# def shuffle_arrays_slow(arrays, seed=None):
#     """SLOWER than shuffle_arrays() !
#     Shuffles copies of arrays in the same order, along axis=0

#     Parameters:
#     -----------
#     arrays : List of NumPy arrays.
#     seed : Seed value if int >= 0, else seed is random.
#     """

#     if seed is not None:
#         np.random.seed(seed)

#     shuffled_arrays = [np.copy(array) for array in arrays]
#     for arr in shuffled_arrays:
#         np.random.shuffle(arr)
    
#     return shuffled_arrays

def split_given_size(a, size):
    return np.split(a, np.arange(size, len(a), size))

def one_hot(Y : np.ndarray, col_wise=False) -> Tuple[np.ndarray, np.ndarray]:
    classes, class_num = np.unique(Y, return_inverse=True)
    # print(classes)
    a = np.eye(len(classes))[class_num].astype('uint8')
    return a.T if col_wise else a, classes

def normalize(data):
    data_norm = (data - np.min(data, axis=0))/ (np.max(data, axis=0) - np.min(data, axis=0))
    return data_norm

def standardize(data):
    standardized_data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    return standardized_data

def split_dataset(dataset: pd.DataFrame, fraction=0.5, seed=None):
    if seed is not None:
        train_set = dataset.sample(frac=fraction, random_state=seed)
    else:
        train_set = dataset.sample(frac=fraction)
    validation_set = dataset.drop(train_set.index)
    return train_set, validation_set
