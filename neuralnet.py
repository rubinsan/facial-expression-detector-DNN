"""
TODO: document this module
"""

import numpy as np
from math_functions import relu, softmax

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
        cache = {}

        # for epoch in epoch_n:
            # forward_prop
            # compute loss
            # backward_prop
            # update parameters
    
    def forward_prop(self, X, Weigths, biases):
        """
        Forward propagation through the network.

        Parameters:
        X -- input features
        Weigths -- dictionary containing the weights matrices
        biases -- dictionary containing the biases vectors

        Returns:
        A -- output of the last layer after activation
        cache -- dictionary containing intermediate values for backpropagation
        """

        for layer in range(len(self.dims) - 1):
            #self.dims[layer+1]
            pass
        

    def compute_loss(self, Z_last, Y):
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
        loss = np.sum(loss_vector) / Y.shape[1]

        return loss
        
    def backward_prop(self, cache, Y):
        """
        Backward propagation through the network to compute gradients.

        Parameters:
        Y -- target labels
        cache -- dictionary containing the cache data from forward propagation

        Returns:
        gradients -- dictionary containing gradients for weights and biases
        """
        gradients = {}
        
        #first dZ=Y_hat-Y
        #first dA -> study

        
        return gradients


    def predict(self, X, Y):
        pass

