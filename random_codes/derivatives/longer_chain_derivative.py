import numpy as np 
from typing import Callable
import matplotlib.pyplot as plt
from plot import *


def derivative(function,ndarray,delta=0.001):
    """[summary]
    
    Arguments:
        function {[]} -- [description]
        ndarray {[numpy array]} -- [input array]
        
    Keyword Arguments:
        delta {float} -- [small change] (default: {0.001})
    
    Returns:
        [constant value] -- [output of derivative function]
    """
    return (function(ndarray + delta)-function(ndarray -delta)/2*delta)


def longer_chain_derivative(list_of_function , ndarray):
    """
    F(x) = f(g(h(x)))
    
    F'(x) = f'(g(h(x))) * g'(h(x)) * h'(x)
    
    Arguments:
        list_of_function {[list]} -- [list of functions]
        ndarray {[numpy array]} -- [input array]
    """
    assert len(list_of_function) == 3, "Number of functions should be 3 "
    assert ndarray.ndim == 1, "Input array should be one dimensional"
    
    f,g,h = list_of_function[0] ,list_of_function[1], list_of_function[2]
    
    #h(x)
    h_of_x = h(ndarray)
    
    #g(h(x))
    g_of_hx = g(h_of_x)
    
    #h'(x) = dh(x)/dx
    dhdx = derivative(h,ndarray)

    #g'(h(x))
    dgdx = derivative(g,h_of_x)

    #f'(g(h(x)))
    dfdx = derivative(f,g_of_hx)
    
    return dfdx * dgdx * dhdx
    
    
##### TEST : sigmoid,leaky relu,square

def square(x):
    return np.power(2,x)

def relu(x):
    return np.maximum(x,0)

def leaky_relu(x):
    return np.maximum(0.2*x,x)

def sigmoid(x):
    return (1/(1+np.exp(-x)))


longer_chain_derivative(list_of_function=[sigmoid,square,leaky_relu],ndarray)
