import numpy as np
from typing import Tuple

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
    print(classes)
    a = np.eye(len(classes))[class_num].astype('uint8')
    return a.T if col_wise else a, classes
