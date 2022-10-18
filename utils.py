from typing import Tuple
import numpy as np
import pandas as pd

# def timeline_sample_faster(series, num):
#     arr = np.random.sample((50, 5000))
#     length = arr.shape[1]
#     for _ in range(num):
#         yield series[: , np.random.permutation(length)]

def shuffle_arrays(arrays, seed=None):
    """Shuffles copies of arrays in the same order, along axis=0

    Parameters:
    -----------
    arrays : List of NumPy arrays.
    seed : Seed value if int >= 0, else seed is random.
    """
    assert all(len(arr) == len(arrays[0]) for arr in arrays)
    seed = np.random.randint(0, 2**(32 - 1) - 1) if seed is None else seed

    shuffled_arrays = [np.copy(array) for array in arrays]
    for arr in shuffled_arrays:
        rstate = np.random.RandomState(seed)
        rstate.shuffle(arr)
    
    return (arr for arr in shuffled_arrays)

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
