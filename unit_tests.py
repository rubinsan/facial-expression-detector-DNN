"""
Scrit for testing the neuralnet class methods
"""
import numpy as np
from neuralnet import NeuralNet
from math_functions import softmax, relu

# Test softmax function
"""
n_classes = 4
batch_size = 6
Z = np.random.randn(n_classes, batch_size)
print(Z)
Y_hat = softmax(Z)
print("Softmax output:")
print(Y_hat)
"""

# Test relu activation function
"""
n_neurons = 4
batch_size = 6
Z = np.random.randn(n_neurons, batch_size)
print(Z)
A = relu(Z)
print("ReLU activation output:")
print(A)
"""

# Test initialization of NeuralNet class
"""
test_NN = NeuralNet([2, 3, 3, 4, 2])
test_NN.initialization()
#for layer in test_NN.weights.keys():
 #   print(layer+":")
  #  print(test_NN.weights[layer])
#for layer in test_NN.biases.keys():
 #   print(layer+":")
  #  print(test_NN.biases[layer])
"""

# Test compute_loss of NeuralNet class
"""
test_NN = NeuralNet([14, 10, 12, 8, 6])
n_classes = test_NN.dims.pop()
batch_size = 5
Z = np.random.randn(n_classes, batch_size)
print(Z)
Y = np.eye(n_classes)[np.random.choice(n_classes, batch_size)]
Y = Y.T
print(Y)
loss = test_NN.compute_loss(Z, Y)
print("loss output:", loss)
"""