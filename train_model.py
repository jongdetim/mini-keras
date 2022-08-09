# %%
import numpy as np
import pandas as pd

# from dense_neural_net import DenseNeuralNet
from Layers import Dense
from activation_functions import *

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
x = np.arange(0, 30, 1, dtype=int)
print(x)

# %%
# model = DenseNeuralNet(dims)

# %%
# print(model._forward_propagation(x, model._sigmoid))

def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    return data

# %%
layer = Dense((4, 2), activation=SoftMax, init_method='xavier', distribution='normal')

#%%
layer.weights

#%%
layer._forward([1, 2, 1, 2.5], activation=SoftMax)
# %%
