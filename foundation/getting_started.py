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
    """
    docstring here
        :param x: 
    """
    
    return (1/1+np.exp(-x))


def derv_relu(value):
    """
    docstring here
        :param value: 
    """
    if value>0:
        return 1
    else:
        return 0 

