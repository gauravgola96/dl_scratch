import numpy as np 
'''
Prefer readme first
'''

def matmul_backward_first(X, W):
    '''
    Computes the backward pass of a matrix multiplication with respect to the
    first argument.
    '''

    # backward pass
    dNdX = np.transpose(W, (1, 0))

    return dNdX


# N = v(X,W) #dot product of X,W
#The dNdX quantity computed here represents the partial derivative of each element of X with respect to the sum of the output N

def matmul_backward_second(X,W):
    '''
    Computes the backward pass of a matrix multiplication with respect to the
    second argument.
    '''

    # backward pass
    dNdW = np.transpose(X, (1, 0))

    return dNdW




