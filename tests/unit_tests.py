"""
Scrit for testing the neuralnet class methods
"""
import numpy as np
from neuralnet_pkg.neuralnet import NeuralNet
from neuralnet_pkg.math_pkg.math_functions import softmax, relu, relu_derivative

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
for layer in test_NN.weights.keys():
    print(layer+":")
    print(test_NN.weights[layer])
for layer in test_NN.biases.keys():
    print(layer+":")
    print(test_NN.biases[layer])
"""

# Test compute_loss method of NeuralNet class
"""
test_NN = NeuralNet([14, 10, 12, 8, 6])
n_classes = test_NN.dims.pop()
batch_size = 5
Z = np.random.randn(n_classes, batch_size)
print(Z)
Y = np.eye(n_classes)[np.random.choice(n_classes, batch_size)]
Y = Y.T
print(Y)
loss = test_NN.compute_cost(Z, Y)
print("loss output:", loss)
"""

# Test forward_prop of NeuralNet class
"""
test_NN = NeuralNet([4, 3, 3, 4, 8])
test_NN.initialization()
batch_size = 4
X = np.random.rand(test_NN.dims[0], batch_size) * 100
print("X:")
print(X)
Z_last, cache = test_NN.forward_prop(X)
for i in range(len(test_NN.dims) - 1):
    print("W"+str(i+1)+":")
    print(test_NN.weights["W"+str(i+1)])
    if (i+1) >= (len(test_NN.dims) - 1): continue
    print("Z"+str(i+1)+":")
    print(cache["Z"+str(i+1)])
    print("A"+str(i+1)+":")
    print(cache["A"+str(i+1)])
"""

# Test of relu_derivative function
"""
test_NN = NeuralNet([4, 3, 3, 4, 5])
test_NN.initialization()
batch_size = 4
X = np.random.rand(test_NN.dims[0], batch_size) * 100
print("X:")
print(X)
Z_last, cache = test_NN.forward_prop(X)
for i in range(len(test_NN.dims) - 1):
    print("Z"+str(i+1)+":")
    print(cache["Z"+str(i+1)])
    print("dG"+str(i+1)+":")
    print(relu_derivative(cache["Z"+str(i+1)]))
"""

# Test backward_prop of NeuralNet class
"""
test_NN = NeuralNet([6, 6, 4, 5, 6])
test_NN.initialization()
batch_size = 4
X = np.random.rand(test_NN.dims[0], batch_size) * 100
print("X:")
print(X)
Z_last, cache = test_NN.forward_prop(X)
Y = np.eye(test_NN.dims[-1])[np.random.choice(test_NN.dims[-1], batch_size)]
Y = Y.T
loss, Y_hat = test_NN.compute_cost(Z_last, Y)
grads = test_NN.backward_prop(cache, Y, Y_hat)
for i in reversed(range(len(test_NN.dims) - 1)):
    print("dW"+str(i+1)+":")
    print(grads["dW"+str(i+1)])
    print("db"+str(i+1)+":")
    print(grads["db"+str(i+1)])
"""

# Test training of NeuralNet class
"""
test_NN = NeuralNet([20, 14, 12, 10, 6])
test_NN.initialization()
batch_size = 1000
X = np.random.rand(test_NN.dims[0], batch_size) * 100
print("X:")
print(X)
Y = np.eye(test_NN.dims[-1])[np.random.choice(test_NN.dims[-1], batch_size)]
Y = Y.T
test_NN.train(X, Y, epoch_n=1000, learning_rate=0.01)
"""

# Test predict of NeuralNet class

test_NN = NeuralNet([20, 14, 12, 10, 6])
test_NN.initialization()
batch_size = 1000
X = np.random.rand(test_NN.dims[0], batch_size) * 100
print("X:")
print(X)
Y = np.eye(test_NN.dims[-1])[np.random.choice(test_NN.dims[-1], batch_size)]
Y = Y.T
test_NN.train(X, Y, epoch_n=1000, learning_rate=0.01)
Y_hat, accuracy = test_NN.predict(X, Y)
print("Test accuracy:", accuracy)

