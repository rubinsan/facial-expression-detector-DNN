"""
Script for testing the neuralnet class library
"""

import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import matplotlib.pyplot as plt
from neuralnet_pkg.neuralnet import NeuralNet
import time

# Dataset import
X_train = np.loadtxt("MNIST_data/mnist_train_data.csv", 
                     delimiter=",", ndmin=2).astype(np.int64).T
Y_train = np.loadtxt("MNIST_data/mnist_train_labels.csv", 
                     delimiter=",").astype(np.int64).T
X_test = np.loadtxt("MNIST_data/mnist_test_data.csv", 
                     delimiter=",", ndmin=2).astype(np.int64).T
Y_test = np.loadtxt("MNIST_data/mnist_test_labels.csv", 
                     delimiter=",").astype(np.int64).T
#X_train = X_train[:, :10000]
#Y_train = Y_train[:10000]
#X_test = X_test[:, :1000]
#Y_test = Y_test[:1000]

# One-hot encoding of labels
Y_train_onehot = np.zeros((Y_train.max() + 1, Y_train.size), dtype=np.int8)
Y_train_onehot[Y_train, np.arange(Y_train.size)] = 1
Y_test_onehot = np.zeros((Y_test.max() + 1, Y_test.size), dtype=np.int8)
Y_test_onehot[Y_test, np.arange(Y_test.size)] = 1

#print(Y_train[:10])  
#print(Y_train_onehot[:, :10])  
Y_train = Y_train_onehot
Y_test = Y_test_onehot
print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)
print("X_test shape:", X_test.shape)
print("Y_test shape:", Y_test.shape)

""" Nerual Network initialization and training """

# hyperparameters
epoch_n = 20
learning_rate = 0.01
decay_rate = 0
mini_batch_size = 256
# Initialize the neural network
test_NN = NeuralNet([X_train.shape[0], 256, 256, Y_train.shape[0]])
test_NN.initialization()
costs = test_NN.train(X_train, Y_train, epoch_n, 
                      learning_rate, mini_batch_size, decay_rate)

Y_hat, accuracy = test_NN.predict(X_train, Y_train)
print("Train accuracy (bias):", accuracy)
Y_hat, accuracy = test_NN.predict(X_test, Y_test)
print("Test accuracy (variance):", accuracy)

# Plotting the training results
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('epoch')
plt.title("Learning rate =" + str(learning_rate))
plt.show()

plt.ion() # turn on interactive mode
plt.show()
for i in range(10):
    idx = np.random.randint(0, Y_test.shape[1], dtype=int)
    Z_last, _ = test_NN.forward_prop(X_test[:, idx:idx+1])
    _, Y_hat = test_NN.compute_cost(Z_last, Y_test[:, idx:idx+1]) 
    a_pred = np.argmax(Y_hat, axis=0)
    plt.figure()
    plt.imshow(X_test[:, idx].reshape((28,28)),cmap='gray')
    plt.title("The model says it is: " + str(a_pred), 
              fontdict = {'fontsize' : 20, 'fontweight' : 10})
    plt.pause(40)
    plt.close() 



