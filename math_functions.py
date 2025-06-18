"""
Math functions definition for neural networks.
"""
import numpy as np

def relu(x):
    return x * (x > 0)

def softmax(Z):
    """Compute the softmax of matrix Z. """   
    exp_Z = np.exp(Z)  
    sm_matrix = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
    return sm_matrix
