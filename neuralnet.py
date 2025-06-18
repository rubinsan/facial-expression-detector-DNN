import numpy as np

class NeuralNet: 
    """
    A class to represent a neural network.
    """
    def __init__(self, *args):
        """
        Initialize the neural network with given layer dimensions.
        
        Arguments:
        *args -- dimensions of each layer
        """
        self.dims = args

    def initialization(*args):
        """
        Parameter initialization for each layer in the network. 
        The first layer is the input feature layer.

        Arguments:
        *args -- dimensions of each layer

        Return:
        W -- dictionary containing the weights matrices
        b -- dictionary containing the biases vectors
        """
        W = {}
        b = {}
        for i in range(len(args) - 1):
            W["i+1"] = np.random.randn(args[i+1], args[i]) * 0.01
            b["i+1"] = np.zeros((args[i+1], 1))

        return W, b

    def forward_prop ():
        pass

