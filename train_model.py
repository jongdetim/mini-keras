# %%
import sys
import numpy as np
import pandas as pd

from dense_neural_net import DenseNeuralNet

# %% constants
dataset_path = 'datasets/data-multilayer-perceptron.csv'
labels = ['id', 'diagnosis', 'mean radius', 'mean texture', 'mean perimiter', 'mean area', 'mean smoothness', 'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal simension',
          'SE radius', 'SE texture', 'SE perimiter', 'SE area', 'SE smoothness', 'SE compactness', 'SE concavity', 'SE concave points', 'SE symmetry', 'SE fractal simension',
          'worst radius', 'worst texture', 'worst perimiter', 'worst area', 'worst smoothness', 'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal simension']

# %% read file
dataset = pd.read_csv(dataset_path, names=labels)

# %%
dataset.head()

# %%
# dims = np.array([30, 10, 10, 2], dtype=int)
dims = [30, 10, 10, 2]
model = DenseNeuralNet(dims)

def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """adds labels and cleans up data

    Args:
        data (pd.DataFrame)

    Returns:
        pd.DataFrame
    """
    return data

# %%
