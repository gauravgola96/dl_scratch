"""
Batch normalization enables the use of higher learning rates. How?

It is more the covariate shift(It is the title of Batch norm original paper) :  change in the distribution of the input values to a learning algorithm

Its success led to use of methods like layer normalization, weights normalization

"""


import numpy as np 

#Refer batch_norm.png


#gamma , beta are initiated with normal dist...
#During backprop they will learn to normalize it better which makes it more powerdful

def batch_norm(X_ndarray, gamma, beta, eps=0.01):
    """[summary]
    
    Arguments:
        X_ndarray {[type]} -- [description]
        gamma {[type]} -- [description]
        beta {[type]} -- [description]
    
    Keyword Arguments:
        eps {float} -- [description] (default: {0.01})
    """
    #Dense layer
    if len(X_ndarray.shape)==2:
        #mean
        mean = np.mean(X_ndarray,axis=0)
        #variance
        variance = np.mean((X_ndarray - mean)**2, axis=0)
        #normalize
        X_hat = (X_ndarray - mean) * 1.0 / np.sqrt(variance + eps)
        
        return gamma * X_hat + beta
    return 'Incorrent input ndarray shape'



