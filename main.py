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

n_samples = 600  # Number of samples to load from each emotion type

X_train, Y_train = load_csv_data('train', n_samples)
print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)

n_samples = 50

X_test, Y_test = load_csv_data('test', n_samples)
print("X_test shape:", X_test.shape)
print("Y_test shape:", Y_test.shape)

""" Nerual Network initialization and training """

# hyperparameters
epoch_n = 10
learning_rate = 0.01
decay_rate = 0
mini_batch_size = 256
# Initialize the neural network
test_NN = NeuralNet([X_train.shape[0], 10000, 10000, Y_train.shape[0]])
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
plt.xlabel('epochs (per hundreds)')
plt.title("Learning rate =" + str(learning_rate))
plt.show()