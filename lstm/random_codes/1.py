import numpy as np

## y = x^2 ###
def x_square(x):
    return np.square(x)

## y = max(x,0)##
def relu(x):
    '''
    Apply "ReLU" function to each element in ndarray.
    '''
    return np.maximum(x, 0)


def leaky_relu(x):
    '''
    Apply "Leaky ReLU" function to each element in ndarray.
    '''
    return np.maximum(0.2 * x, x)

def sigmoid(x):
    return (1/1+np.exp(-x))


