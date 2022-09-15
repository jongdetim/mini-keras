#%%
%load_ext autoreload
%autoreload 2

# %%
from loss_functions import *
from activation_functions import *
from models import Sequential
from layers import Dense
import pandas as pd
import numpy as np

# %%

# from dense_neural_net import DenseNeuralNet

# #%% TESTING
# model = Sequential([Dense((2, 3), Tanh),
#                     Dense((3, 11), Tanh),
#                     Dense((11, 2), SoftMax)], BinaryCrossEntropy)

# #%% TEST [2, 1] logistic regression
# model = Sequential([Dense((2, 1), Sigmoid)], BinaryCrossEntropy)

# %% TEST - IT LEARNS SLOWER WITH ACTIVATED LAYERS! why?
model = Sequential([Dense((3, 5), activation=ReLU),
                    Dense((5, 2), activation=SoftMax)], BinaryCrossEntropy)

# #%% TEST input + 2 layers
# model = Sequential([Dense((2, 4), Sigmoid),
#                     Dense((4, 2), Sigmoid)], BinaryCrossEntropy)
# # why does the additional layer converge slower than logistic regression?
# # -> because sigmoid layers reduce convergence rate

# #%% TEST input + 2 BIG layers
# model = Sequential([Dense((2, 10), ReLU),
#                     Dense((10, 20), ReLU),
#                     Dense((20, 30), ReLU),
#                     Dense((30, 40), ReLU),
#                     Dense((40, 2), SoftMax)], BinaryCrossEntropy)


# #%% TEST input + 2 layers & 2 output
# model = Sequential([Dense((2, 4), Sigmoid),
#                     Dense((4, 2), Sigmoid)], BinaryCrossEntropy)

# #%% TEST many layers
# model = Sequential([Dense((2, 2), Sigmoid),
#                     Dense((2, 2), Sigmoid),
#                     Dense((2, 2), Sigmoid),
#                     Dense((2, 2), Sigmoid),
#                     Dense((2, 2), Sigmoid),
#                     Dense((2, 2), SoftMax)], BinaryCrossEntropy)

# %% test data
x = np.array([np.array([0.3, 0.4, 1.3]).reshape(-1, 1),
             np.array([0.9, 0.5, 0.3]).reshape(-1, 1)])
# x = np.array([[0.3, 0.4, 1.3], [0.3, 0.4, 1.3], [0.3, 0.4, 1.3], [0.3, 0.4, 1.3]])
x = np.array([[0.3, 0.4, 1.3]])
y = np.array([np.array([1, 0]).reshape(-1, 1),
             np.array([0, 1]).reshape(-1, 1)])
# y = np.array([[1, 0], [1, 0], [1, 0], [1, 0]])
y = np.array([[1, 0]])
print(x, y)
print(x.shape, y.shape)

# %%
model.fit(x, y, epochs=500, learning_rate=0.01, stochastic=False)

# %%
for layer in model.layers:
    print("weights:", layer.weights)
    print("biases:", layer.biases)

# %%
v = np.array([-1.0, -1.0, 1.0])
truth = np.array([0, 1, 0])
soft = SoftMax.forward(v)
BinaryCrossEntropy.forward(soft, truth)

# %%
inputt = np.array([1, 2]).reshape(-1, 1)
out = SoftMax.forward(inputt)
truth = np.array([0, 1]).reshape(-1, 1)

loss = BinaryCrossEntropy.forward(out, truth)
print(SoftMax.backward(inputt, BinaryCrossEntropy.backward(out, truth)))
print(out - truth)

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
layer = Dense((4, 2), activation=SoftMax,
              init_method='xavier', distribution='normal')

# %%
layer.weights

# %%
layer.forward([1, 2, 1, 2.5], activation=SoftMax)
# %%
