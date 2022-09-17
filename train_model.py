#%%
%load_ext autoreload
%autoreload 2

import pandas as pd
import numpy as np
from loss_functions import *
from activation_functions import *
from models import Sequential
from layers import Dense
from typing import Tuple

dataset_path = 'datasets/data-multilayer-perceptron.csv'
labels = ['id', 'diagnosis', 'mean radius', 'mean texture', 'mean perimiter', 'mean area', 'mean smoothness', 'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal simension',
          'SE radius', 'SE texture', 'SE perimiter', 'SE area', 'SE smoothness', 'SE compactness', 'SE concavity', 'SE concave points', 'SE symmetry', 'SE fractal simension',
          'worst radius', 'worst texture', 'worst perimiter', 'worst area', 'worst smoothness', 'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal simension']

def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    return data

def one_hot(Y : np.ndarray, col_wise=False) -> Tuple[np.ndarray, np.ndarray]:
    classes, class_num = np.unique(Y, return_inverse=True)
    print(classes)
    a = np.eye(len(classes))[class_num].astype('uint8')
    return a.T if col_wise else a, classes

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    pass

def normalize_data(data):
    data_norm = (data-data.min())/ (data.max() - data.min())
    return data_norm

# %% read file
dataset = pd.read_csv(dataset_path, names=labels)

# %%
dataset.head()

#%%
Y, Y_labels = one_hot(dataset['diagnosis'].to_numpy())
X = dataset[['mean radius', 'mean texture']].to_numpy()
X = normalize_data(X)
print(X)



# %%
model = Sequential([Dense((2, 5), activation=ReLU),
                    Dense((5, 2), activation=SoftMax)], BinaryCrossEntropy)

# model.layers[0].weights = np.array([[-0.5, -0.3, -0.1, -1.3, -1],
#                              [-0.2, -0.1, -0.8, -0.9, -1.8]]).T
# model.layers[1].weights = np.array([[-0.5, -0.3, -0.1, -1.3, -1],
#                              [-0.2, -0.1, -0.8, -0.9, -1.8]])

# %%
model.fit(X, Y, epochs=100, learning_rate=0.01, stochastic=False)

# %%
