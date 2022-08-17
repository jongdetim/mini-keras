#%%
%load_ext autoreload
%autoreload 2

# %%
import numpy as np
import pandas as pd

# from dense_neural_net import DenseNeuralNet
from layers import Dense
from models import Sequential
from activation_functions import *
from loss_functions import *

#%% TESTING
model = Sequential([Dense((2, 3), ReLU),
                    Dense((3, 11), Sigmoid),
                    Dense((11, 2), SoftMax)], BinaryCrossEntropy)

#%%
model.fit(np.array([3, 15]), np.array([0, 1]))

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
layer.forward([1, 2, 1, 2.5], activation=SoftMax)
# %%
