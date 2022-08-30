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
model = Sequential([Dense((2, 3), Tanh),
                    Dense((3, 11), Tanh),
                    Dense((11, 2), SoftMax)], BinaryCrossEntropy)

#%% TEST [2, 1] logistic regression
model = Sequential([Dense((2, 2), Sigmoid)], BinaryCrossEntropy)

#%% TEST - IT LEARNS SLOWER WITH ACTIVATED LAYERS! why?
model = Sequential([Dense((2, 1), activation=ReLU),
                    Dense((1, 1), activation=ReLU),
                    Dense((1, 1), activation=Sigmoid)], BinaryCrossEntropy)

#%% TEST input + 2 layers
model = Sequential([Dense((2, 4), Sigmoid),
                    Dense((4, 2), Sigmoid)], BinaryCrossEntropy)
# why does the additional layer converge slower than logistic regression?
# -> because the data is easily linearly separable. less noisy parameters

#%% TEST input + 2 BIG layers
model = Sequential([Dense((2, 10), ReLU),
                    Dense((10, 20), ReLU),
                    Dense((20, 30), ReLU),
                    Dense((30, 40), ReLU),
                    Dense((40, 2), SoftMax)], BinaryCrossEntropy)
# bigger model learns faster

#%% TEST input + 2 layers & 2 output
model = Sequential([Dense((2, 4), Sigmoid),
                    Dense((4, 2), Sigmoid)], BinaryCrossEntropy)

#%% TEST many layers
model = Sequential([Dense((2, 2), Sigmoid),
                    Dense((2, 2), Sigmoid),
                    Dense((2, 2), Sigmoid),
                    Dense((2, 2), Sigmoid),
                    Dense((2, 2), Sigmoid),
                    Dense((2, 2), SoftMax)], BinaryCrossEntropy)
# SoftMax is malfunctioning

#%% 
model.fit(np.array([20, -10]).reshape(-1, 1), np.array([1]).reshape(-1, 1), epochs=500, learning_rate=0.01)

#%%
for layer in model.layers:
    print("weights:", layer.weights)
    print("biases:", layer.biases)

#%%
SoftMax.backward(np.array([0.3, 0.5]), np.array([0.4, -0.6]))

#%%
SoftMax.backward2(np.array([0.3, 0.5]), np.array([0.4, -0.6]))

#%%
SoftMax.backward3(np.array([0.3, 0.5]), np.array([0.4, -0.6]))

#%%
SoftMax.backward4(np.array([0.3, 0.5]).reshape(-1, 1), np.array([0.4, 0.9]).reshape(-1, 1))

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
