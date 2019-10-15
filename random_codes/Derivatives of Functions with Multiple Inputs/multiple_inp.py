import numpy as np 


def derivative(ndarray,function,delta=0.01):
    return (function(ndarray+delta)-function(ndarray-delta)/2*delta)


#Passing multiple inputs to a function
 
def multiple_input_to_function(x, y, function):
    """
     Arguments:
         x {[type]} -- [one darray]
         y {[type]} -- [one darray]
         function {[type]} -- [function which takes multiple input]
    """
    assert x.shape == y.shape, "Shape of x and y should be same but here shape of x is {} and y is {}".format(x.shape,y.shape)
    
    return function(x,y)


##Derivatives of Functions with Multiple Inputs

def multiple_inp_add_forward_backward(x, y, function):
    """
    Computes the derivative of this simple function with respect to
    both inputs.

    Arguments:
        x {[type]} -- [one darray]
        y {[type]} -- [one darray]
        function {[type]} -- [description]
    """
    add = x + y 
    
    #derivative 
    dsda = derivative(ndarray=add,function=function)
    
    dadx , dady = 1 , 1 
    
    return dsda * dadx , dsda * dady

## Dot product 

def matrix_mult_forward(x, y,):
    """
    Arguments:
        x {[type]} -- [ndarray]
        y {[type]} -- [ndarray]
    """
    
    assert x.shape[0] == y.shape[1], "Number of columns in first mat should match with number of rows in second mat"
    
    return np.dot(x,y) 









