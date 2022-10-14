#%% delete magic after development
%load_ext autoreload
%autoreload 2

import pandas as pd
import numpy as np

from loss_functions import *
from activation_functions import *
from models import Sequential
from layers import Dense
from utils import one_hot, standardize, split_dataset

pd.options.mode.chained_assignment = None

dataset_path = 'datasets/data-multilayer-perceptron.csv'
labels = ['id', 'diagnosis', 'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness', 'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal simension',
          'SE radius', 'SE texture', 'SE perimeter', 'SE area', 'SE smoothness', 'SE compactness', 'SE concavity', 'SE concave points', 'SE symmetry', 'SE fractal simension',
          'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness', 'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal simension']

def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    return data


# %% read file
full_dataset = pd.read_csv(dataset_path, names=labels)

#%%
dataset = full_dataset[['worst area', 'worst smoothness', 'mean texture', 'diagnosis']]
Y, Y_labels = one_hot(dataset['diagnosis'].to_numpy())
dataset = dataset.drop('diagnosis', axis=1)
dataset = standardize(dataset)
dataset['Y'] = Y.tolist()

train_set, validation_set = split_dataset(dataset, 0.5)

X_train = train_set[['worst area', 'worst smoothness', 'mean texture']].to_numpy()
X_validation = validation_set[['worst area', 'worst smoothness', 'mean texture']].to_numpy()
Y_train = np.vstack(train_set['Y'].to_numpy())
Y_validation = np.vstack(validation_set['Y'].to_numpy())

validation_set = {'X': X_validation, 'Y': Y_validation}

#%%
# x_ = dataset[['worst area', 'worst smoothness', 'mean texture']].to_numpy()
# X = standardize(X)

# %%
model = Sequential([Dense((3, 2), activation=SoftMax)], BinaryCrossEntropy)

# %%
model = Sequential([Dense((3, 5), activation=ReLU),
                    Dense((5, 2), activation=SoftMax)], BinaryCrossEntropy)

# %%
model = Sequential([Dense((3, 2), activation=ReLU),
                    Dense((2, 1), activation=ReLU),
                    Dense((1, 2), activation=SoftMax)], BinaryCrossEntropy)


#%%
model.fit(X_train, Y_train, epochs=500, learning_rate=0.01, batch_size=32, validation_set=validation_set)
model.score(X_validation, Y_validation)

#%%
model.predict(X_validation, Y_labels, output_type='exclusive')
# model.predict(X[0:5], Y_labels, output_type='numerical')

#%%
# model.loss(X, Y)

# #%%
# model.score(X, Y)

# %%
for layer in model.layers:
    print("weights:", layer.weights)
    print("biases:", layer.biases)

# %%
