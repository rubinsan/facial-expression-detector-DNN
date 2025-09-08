"""
Script for testing the neuralnet class library with MNIST dataset
Author: Ruben Sanchez - RubinSan
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
#X_train = X_train[:, :5000]
#Y_train = Y_train[:5000]
#X_test = X_test[:, :500]
#Y_test = Y_test[:500]

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

# """ several Nerual Networks initialization, training and evaluation """
# # hyperparameters
# epoch_n = 25
# learning_rate = 0.01
# decay_rate = 0
# mini_batch_size = 128

# test_NN = NeuralNet([X_train.shape[0], 256, 256, Y_train.shape[0]])
# test_NN.initialization()
# costs = test_NN.train(X_train, Y_train, epoch_n, 
#                       learning_rate, mini_batch_size, decay_rate)
# plt.plot(costs, label='mini-batch size = ' + str(mini_batch_size) +
#          ", rate decay =" + str(decay_rate))

# epoch_n = 25
# learning_rate = 0.01
# decay_rate = 1
# mini_batchs = [128, 32, 16]

# # loop over different mini-batch sizes
# for mini_batch_size in mini_batchs:
#     test_NN = NeuralNet([X_train.shape[0], 256, 256, Y_train.shape[0]])
#     test_NN.initialization()
#     costs = test_NN.train(X_train, Y_train, epoch_n, 
#                         learning_rate, mini_batch_size, decay_rate)

#     #Y_hat, accuracy = test_NN.predict(X_train, Y_train)
#     #print("Train accuracy (bias):", accuracy)
#     #Y_hat, accuracy = test_NN.predict(X_test, Y_test)
#     #print("Test accuracy (variance):", accuracy)

#     # Plotting the training results
#     plt.plot(costs, label='mini-batch size = ' + str(mini_batch_size) +
#             ", rate decay =" + str(decay_rate))

# #X_train = X_train / 255.0
# #Y_train = Y_train / 255.0
# #test_NN = NeuralNet([X_train.shape[0], 256, 256, Y_train.shape[0]])
# #test_NN.initialization()
# #costs = test_NN.train(X_train, Y_train, epoch_n, 
# #                      learning_rate, mini_batch_size, decay_rate)

# plt.ylabel('cost')
# plt.xlabel('epoch')
# plt.title("Learning rate =" + str(learning_rate) + 
#           " with rate decay implementation")
# plt.legend(loc="upper right")
# plt.show()

""" JUST 1 Nerual Network initialization, training and evaluation """

# hyperparameters
epoch_n = 25
learning_rate = 0.01
decay_rate = 0
mini_batch_size = 128
# Initialize the neural network
test_NN = NeuralNet([X_train.shape[0], 256, 256, Y_train.shape[0]])
test_NN.initialization()
## Training and evaluation of accuracy at each epoch
# train_accuracy = []
# test_accuracy = []
# for epoch in range(epoch_n):
#     costs = test_NN.train(X_train, Y_train, 1, 
#                       learning_rate, mini_batch_size, decay_rate)
#     Y_hat, accuracy = test_NN.predict(X_train, Y_train)
#     train_accuracy.append(accuracy)
#     print("Train accuracy (bias):", accuracy)
#     Y_hat, accuracy = test_NN.predict(X_test, Y_test)
#     test_accuracy.append(accuracy)
#     print("Test accuracy (variance):", accuracy)

# plt.plot(train_accuracy, label='Train accuracy')
# plt.plot(test_accuracy, label='Test accuracy')
# plt.legend(loc="lower right")
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# #plt.title("Learning rate =" + str(learning_rate))
# plt.show()

# Evaluation of accuracy and training performance
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

# """ Visualizing predictions on test set """

# # plt.ion() # turn on interactive mode
# # plt.show()
# # for i in range(10):
# #     idx = np.random.randint(0, Y_test.shape[1], dtype=int)
# #     Z_last, _ = test_NN.forward_prop(X_test[:, idx:idx+1])
# #     _, Y_hat = test_NN.compute_cost(Z_last, Y_test[:, idx:idx+1]) 
# #     a_pred = np.argmax(Y_hat, axis=0)
# #     plt.figure()
# #     plt.imshow(X_test[:, idx].reshape((28,28)),cmap='gray')
# #     plt.title("The model says it is: " + str(a_pred), 
# #               fontdict = {'fontsize' : 20, 'fontweight' : 10})
# #     plt.pause(40)
# #     plt.close() 



