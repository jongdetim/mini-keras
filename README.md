# Mini-Keras

A small library to create neural networks from scratch, built with numpy. It has a keras-like syntax and options, and is built to be easily extendible to different models and architectures.

The implemented model uses stochastic/batch gradient descent to train a Multilayer Perceptron.

&nbsp;

Example:
```py
from mini_keras.layers import Dense
from mini_keras.models import Sequential
from mini_keras.activation_functions import LReLU, SoftMax
from mini_keras.loss_functions import BinaryCrossEntropy

model = Sequential([Dense((30, 8), activation=LReLU),
                    Dense((8, 5), activation=LReLU),
                    Dense((5, 2), activation=SoftMax)], BinaryCrossEntropy)

model.fit(X, Y)
```

## Documentation

[Documentation](https://jongdetim.github.io/mini-keras)

## Setup

Python 3 is required to use this project

run:
```bash
pip install -r requirements.txt
```
to install dependencies

## Demo

Running:
```
python3 demo_train_mlp.py
```
will execute a demo script that preprocesses the [Breast Cancer Wisconsin (Diagnostic) Data Set](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29), creates a model with two hidden layers, trains the model & saves the model architecture and parameters as a .pickle file. It also displays a graph of the loss during training.

&nbsp;

```
python3 demo_predict_mlp.py ./datasets/data-multilayer-perceptron.csv MLP.pickle
```
runs another demo script that takes a dataset and loads the previously saved model. The loss and accuracy of the model are displayed.

&nbsp;
&nbsp;

Possible improvements:
- Decoupling of some responsibilities in the Sequential class
- GPU support using CUDA
- Other types of layers, such as convolution & pooling layers
- Other types of models, such as sequential networks
- Early stopping
- Regularization
