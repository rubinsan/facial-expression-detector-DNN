"""
Math functions definition for neural networks.
"""
import numpy as np

def relu(Z):
    """Compute the ReLU function of matriz Z. """ 
    return Z * (Z > 0)

def softmax(Z):
    """Compute the softmax of matrix Z. """   
    exp_Z = np.exp(Z)  
    sm_matrix = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
    return sm_matrix

def relu_derivative(Z):
    """Compute the derivative of the ReLU function respect to the matrix Z """
    return (Z >= 0).astype(Z.dtype)

