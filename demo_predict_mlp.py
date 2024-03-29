import sys
import pandas as pd

from mini_keras.models import Sequential
from mini_keras.utils import one_hot, standardize

labels = ['id', 'diagnosis', 'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness', 'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension',
          'SE radius', 'SE texture', 'SE perimeter', 'SE area', 'SE smoothness', 'SE compactness', 'SE concavity', 'SE concave points', 'SE symmetry', 'SE fractal dimension',
          'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness', 'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension']

# %% reading file
if len(sys.argv) <= 2:
    print("please provide: 1. path to a .csv file, and 2. a file containing saved model")
    sys.exit(1)
else:
    dataset_path = sys.argv[1]
    model_path = sys.argv[2]
data = pd.read_csv(dataset_path)

#%%
model = Sequential.load(model_path)

# %% read file
full_dataset = pd.read_csv(dataset_path, names=labels)

#%% preprocess data
dataset = full_dataset.iloc[:,1:32]
Y, Y_labels = one_hot(dataset['diagnosis'].to_numpy())
dataset = dataset.drop('diagnosis', axis=1)
X = standardize(dataset)

#%%
print("cross entropy loss:", model.loss(X, Y))
print("accuracy:", model.score(X, Y))
