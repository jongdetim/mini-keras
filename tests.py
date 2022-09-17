#%%
%load_ext autoreload
%autoreload 2

# %%
from loss_functions import *
from activation_functions import *
import pandas as pd
import numpy as np

#%%
# df = pd.DataFrame({"A" : [0, 2], "B" : [1, 3]})
# # df
# z = df.iloc[0].to_numpy()
# z.shape
# z.reshape(-1, 1).shape
# z.reshape(-1, 1)

#%%
# model = Sequential([Dense((3, 5), activation=Tanh),
#                     Dense((5, 2), activation=SoftMax)], BinaryCrossEntropy)

# %%
# for layer in model.layers:
#     print("weights:", layer.weights)
#     print("biases:", layer.biases)

# %% test data
# x = np.array([np.array([0.3, 0.4, 1.3]).reshape(-1, 1),
#              np.array([0.9, 0.5, 0.3]).reshape(-1, 1)])
# # x = np.array([[0.3, 0.4, 1.3], [0.3, 0.4, 1.3], [0.3, 0.4, 1.3], [0.3, 0.4, 1.3]])
# x = np.array([[0.3, 0.4, 1.3]])
# y = np.array([np.array([1, 0]).reshape(-1, 1),
#              np.array([0, 1]).reshape(-1, 1)])
# # y = np.array([[1, 0], [1, 0], [1, 0], [1, 0]])
# y = np.array([[1, 0]])
# print(x, y)
# print(x.shape, y.shape)

#%%
# t = np.array([[1, 2],[3, 4]])
# # t = np.array([3, 4])
# print(t.reshape(-1, t.shape[0]))
# # t = np.array([3, 4])
# t = np.array([[1, 2],[3, 4]])
# print(t.reshape(t.shape[0], -1))

#%% Cross entropy with softmax tests
x = np.array([[1.7, 3.1]]).T
print(x.shape)
truth = np.array([[0, 1]]).T
prediction = SoftMax.forward(x)
gradient = BinaryCrossEntropy.backward(prediction, truth)
result = SoftMax.backward(x, gradient)
assert(np.allclose(result,prediction - truth)), f"result: {result} is not equal to prediction - truth {prediction-truth}"

#%%
x = np.array([[1.7, 3.1],
             [0.5, -0.2]])
print(x.shape)
truth = np.array([[0, 1],
                 [1, 0]])
prediction = SoftMax.forward(x)
gradient = BinaryCrossEntropy.backward(prediction, truth)
result = SoftMax.backward(x, gradient)
assert(np.allclose(result,prediction - truth)), f"result: {result} is not equal to prediction - truth {prediction-truth}"

# %%
print("all tests passed!")
