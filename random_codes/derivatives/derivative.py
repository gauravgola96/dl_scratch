import numpy as np 
import matplotlib.pyplot as plt


def sqaure(value):
    return(np.power(2,value))


def sigmoid(value):
    """
    Apply sigmoid function to each value of ndarray
    value : ndarray
    """
    return (1+(1+np.exp(-value)))


def derivative(function , ndarray,delta=0.001):
    """
    Apply slope formula for a very small value delta
        :param function: 
        :param ndarray: 
    """
    derv = (function(ndarray+delta) - function(ndarray - delta))/2*delta
    return derv
    

def chain_derivative(function_list, ndarray):
    """
    d (f2(f1(x)))/dx = f2`f1(x) * f1`(x)
    
    Easy notation: F(x) = f(g(x))
    
    F`(x) = f`(g(x)) * g`(x)
    :param function_list: list of function eg. [square,sigmoid]
    :param ndarray: input array
    """
    assert len(function_list) == 2 #Number of functions should be no more than two
    assert ndarray.ndim == 1 #shape of array should be 1 dimensional
    
    f, g = function_list[0], function_list[1]
    
    #g(x)
    g_of_x = g(ndarray)
    
    #g`(x) = dg/dx  
    
    dgdx = derivative(g,ndarray)
    
    #f`(g(x)) = df/dx*g(x)
    dfdx = derivative(f,g(ndarray))
    
    chain_derv =  dfdx * dgdx
    print('shape of derivative {}'.format(chain_derv.shape))
    
    return chain_derv



###### TEST F(x) = sigmoid(square)######

ndarray = np.arange(-3,3,0.01)
function_list = [sigmoid,sqaure]

# print(chain_derivative(function_list=function_list,ndarray=ndarray))
fig = plt.plot(chain_derivative(function_list=function_list,ndarray=ndarray),ndarray)
plt.savefig('chain_derv.png')

