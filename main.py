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


X_train, Y_train = load_csv_data('train')
X_train = X_train[:, :5000]
Y_train = Y_train[:5000]
X_test, Y_test = load_csv_data('test')
X_test = X_test[:, :500]
Y_test = Y_test[:500]

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