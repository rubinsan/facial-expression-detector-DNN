"""
TODO: document this module
"""

import numpy as np
import matplotlib.pyplot as plt
from math_functions import relu, softmax, relu_derivative

class NeuralNet: 
    """
    A class to represent a neural network.
    """
    def __init__(self, nn_dims):
        """Initialize the neural network with given layer dimensions."""
        self.dims = nn_dims
        self.weights = {}
        self.biases = {}

    def initialization(self):
        """
        Parameter initialization for each layer in the network. 
        The first layer is the input feature layer.
        """
        for i in range(len(self.dims) - 1):
            self.weights["W"+str(i+1)] = np.random.randn(self.dims[i+1], 
                                                         self.dims[i]) * 0.01
            self.biases["b"+str(i+1)] = np.zeros((self.dims[i+1], 1))

        return None

    def train (self, X, Y, epoch_n=1000, learning_rate=0.01):
        """
        Train the neural network using the provided data and labels with the
        specified learning rate.

        Parameters:
        X -- input features
        Y -- target labels
        epoch_n -- number of epochs for training
        learning_rate -- learning rate for gradient descent

        Returns:

        """

        for epoch in range(epoch_n):
            Z_last, cache = self.forward_prop(X)
            cost, Y_hat = self.compute_cost(Z_last, Y)
            grads = self.backward_prop(cache, Y, Y_hat)
            self.gradient_descent(grads, learning_rate)
            if epoch % 100 == 0:
                print("Cost after epoch:", epoch, ":", cost)

            """  
            plt.plot(cost)
            plt.ylabel('cost')
            plt.xlabel('epochs (per hundreds)')
            plt.title("Learning rate =" + str(learning_rate))
            plt.show()
            """
    
    def forward_prop(self, X):
        """
        Forward propagation through the network.

        Parameters:
        X -- input sample matrix

        Returns:
        Z_last -- output of the last layer before activation
        cache -- dictionary containing intermediate values for backpropagation
        """
        cache = {}
        Z = []
        A = X
        cache["A0"] = X # store for backpropagation
        for i in range(len(self.dims) - 1):
            W = self.weights["W"+str(i+1)]
            Z = np.matmul(W, A) + self.biases["b"+str(i+1)]
            A = relu(Z)
            if (i+1) >= (len(self.dims) - 1): continue  # last layer not stored
            cache["Z"+str(i+1)] = Z
            cache["A"+str(i+1)] = A 
        Z_last = Z  # Last layer output before activation
        return Z_last, cache

    def compute_cost(self, Z_last, Y):
        """
        Compute the loss using cross-entropy loss function.

        Parameters:
        Z_last -- output matrix of the last layer before activation
        Y -- target labels matrix

        Returns:
        loss -- computed loss value
        """
        Y_hat = softmax(Z_last)
        Y_hat_vector = np.sum((Y*Y_hat), axis=0, keepdims=True)
        loss_vector = -np.log(Y_hat_vector)
        cost = np.sum(loss_vector) / Y.shape[1]

        return cost, Y_hat
        
    def backward_prop(self, cache, Y, Y_hat):
        """
        Backward propagation through the network to compute gradients.

        Parameters:
        Y -- target labels
        cache -- dictionary containing the cache data from forward propagation

        Returns:
        gradients -- dictionary containing gradients for weights and biases
        """
        gradients = {}
        m = 1/Y.shape[1] # number of samples
        
        # last layer calculations, different activation function (softmax)
        dZ = Y_hat-Y # first dZ
        dW = m * np.matmul(dZ, cache["A"+str(len(self.dims)-2)].T)
        db = m * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.matmul(self.weights["W"+str(len(self.dims)-1)].T, dZ)
        gradients["dW"+str(len(self.dims)-1)] = dW
        gradients["db"+str(len(self.dims)-1)] = db
        cache.pop("A"+str(len(self.dims)-2), None)

        for i in reversed(range(len(self.dims) - 2)): # loop from last - 1 layer
            dZ = dA_prev * relu_derivative(cache["Z"+str(i+1)]) 
            dW = m * np.matmul(dZ, cache["A"+str(i)].T)
            db = m * np.sum(dZ, axis=1, keepdims=True)
            dA_prev = np.matmul(self.weights["W"+str(i+1)].T, dZ)
            cache.pop("Z"+str(i+1), None) # release memory
            cache.pop("A"+str(i), None) # release memory
            gradients["dW"+str(i+1)] = dW
            gradients["db"+str(i+1)] = db
            
        return gradients

    def gradient_descent(self, gradients, learning_rate):
        """
        Update the weights and biases using gradient descent.

        Parameters:
        gradients -- dictionary containing gradients for weights and biases
        learning_rate -- learning rate for the update step
        """
        for i in range(len(self.dims) - 1):
            self.weights["W"+str(i+1)] -= learning_rate * gradients["dW"+str(i+1)]
            self.biases["b"+str(i+1)] -= learning_rate * gradients["db"+str(i+1)]
            gradients.pop("dW"+str(i+1), None) # release memory
            gradients.pop("db"+str(i+1), None) # release memory

    def predict(self, X, Y):
        pass

