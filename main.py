"""
Script for testing the neuralnet class library
"""

import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import matplotlib.pyplot as plt
from neuralnet_pkg.neuralnet import NeuralNet
from facial_data_proc_pkg.facial_data_process import jpg_to_csv, load_csv_data

# Dataset import and formatting



X_train = np.loadtxt("MNIST_data/mnist_train_data.csv", 
                     delimiter=",", ndmin=2).astype(np.int64).T
Y_train = np.loadtxt("MNIST_data/mnist_train_labels.csv", 
                     delimiter=",").astype(np.int64).T
X_test = np.loadtxt("MNIST_data/mnist_test_data.csv", 
                     delimiter=",", ndmin=2).astype(np.int64).T
Y_test = np.loadtxt("MNIST_data/mnist_test_labels.csv", 
                     delimiter=",").astype(np.int64).T
#print("X_train shape:", X_train.shape)
#print("Y_train shape:", Y_train.shape)
X_train = X_train[:, :5000]
Y_train = Y_train[:5000]
X_test = X_test[:, :500]
Y_test = Y_test[:500]

Y_train_onehot = np.zeros((Y_train.max() + 1, Y_train.size), dtype=np.int8)
Y_train_onehot[Y_train, np.arange(Y_train.size)] = 1

Y_test_onehot = np.zeros((Y_test.max() + 1, Y_test.size), dtype=np.int8)
Y_test_onehot[Y_test, np.arange(Y_test.size)] = 1

#print(Y_train[:10])  # Display first 10 labels
#print(Y_train_onehot[:, :10])  # Display first 10 one-hot encoded labels
Y_train = Y_train_onehot
Y_test = Y_test_onehot
print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)
print("X_test shape:", X_test.shape)
print("Y_test shape:", Y_test.shape)

""" Nerual Network initialization and training """

# hyperparameters
epoch_n = 1000
learning_rate = 0.01
# Initialize the neural network
test_NN = NeuralNet([X_train.shape[0], 256, 256, Y_train.shape[0]])
test_NN.initialization()
costs = test_NN.train(X_train, Y_train, epoch_n, learning_rate)
# Plotting the training results
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('epochs (per hundreds)')
plt.title("Learning rate =" + str(learning_rate))
plt.show()
Y_hat, accuracy = test_NN.predict(X_test, Y_test)
print("Test accuracy:", accuracy)