import numpy as np
from sklearn.utils import shuffle

def shuffle_arrays(arrays, seed=-1):
    """Shuffles copies of arrays in the same order, along axis=0

    Parameters:
    -----------
    arrays : List of NumPy arrays.
    seed : Seed value if int >= 0, else seed is random.
    """
    assert all(len(arr) == len(arrays[0]) for arr in arrays)
    seed = np.random.randint(0, 2**(32 - 1) - 1) if seed < 0 else seed

    shuffled_arrays = np.copy(arrays)
    for arr in shuffled_arrays:
        rstate = np.random.RandomState(seed)
        rstate.shuffle(arr)
    
    return (arr for arr in shuffled_arrays)

def split_given_size(a, size):
    return np.split(a, np.arange(size, len(a), size))
