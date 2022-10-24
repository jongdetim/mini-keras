#%% delete magic after development
# %load_ext autoreload
# %autoreload 2

import pandas as pd
import numpy as np

from loss_functions import *
from activation_functions import *
from models import Sequential
from layers import Dense
from utils import one_hot, standardize, split_dataset

pd.options.mode.chained_assignment = None

dataset_path = 'datasets/data-multilayer-perceptron.csv'
labels = ['id', 'diagnosis', 'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness', 'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension',
          'SE radius', 'SE texture', 'SE perimeter', 'SE area', 'SE smoothness', 'SE compactness', 'SE concavity', 'SE concave points', 'SE symmetry', 'SE fractal dimension',
          'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness', 'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension']

def print_full(x):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    pd.set_option('display.float_format', '{:20,.2f}'.format)
    pd.set_option('display.max_colwidth', None)
    print(x.info())
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.float_format')
    pd.reset_option('display.max_colwidth')

# %% read file
full_dataset = pd.read_csv(dataset_path, names=labels)

#%% preprocess data
dataset = full_dataset.iloc[:,1:32]
Y, Y_labels = one_hot(dataset['diagnosis'].to_numpy())
dataset = dataset.drop('diagnosis', axis=1)
dataset = standardize(dataset)
dataset['Y'] = Y.tolist()

train_set, validation_set = split_dataset(dataset, 0.5, seed=1)

X_train = train_set.iloc[:,:-1].to_numpy()
X_validation = validation_set.iloc[:,:-1].to_numpy()
Y_train = np.vstack(train_set['Y'].to_numpy())
Y_validation = np.vstack(validation_set['Y'].to_numpy())

validation_set = {'X': X_validation, 'Y': Y_validation}

# %%
model = Sequential([Dense((30, 8), activation=LReLU, seed=1),
                    Dense((8, 5), activation=LReLU, seed=1),
                    Dense((5, 2), activation=SoftMax, seed=1)], BinaryCrossEntropy)

#%%
model.fit(X_train, Y_train, epochs=700, learning_rate=0.01, batch_size=32, validation_set=validation_set, seed=1)
print("training set", model.score(X_train, Y_train))
print("validation set", model.score(X_validation, Y_validation))

#%%
model.save("MLP", as_json=False)
