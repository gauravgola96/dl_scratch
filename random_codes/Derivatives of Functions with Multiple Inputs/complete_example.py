import numpy as np 

'''
Example : example.png
Let’s suppose our function takes in the vectors X and W,
 performs the dot product described in the prior section—which we’ll 
 denote as ν ( X , W ) —and then feeds the vectors through a function σ.

s = f ( X , W ) = σ ( ν ( X , W ) ) = σ ( x 1 × w 1 + x 2 × w 2 + x 3 × w 3 )
'''


def sigmoid(value):
    return (1/1+np.exp(-value))

def derivative_sigmoid(value):
    return sigmoid(value)(1-sigmoid(value))

def derivative(ndarray, function, delta=0.01):
    return (function(ndarray+delta) - function(ndarray - delta)/2*delta)

def forward_matrix(x,w,function=sigmoid):
    """
    Arguments:
        x {[type]} -- [description]
        w {[type]} -- [description]
        function {[type]} -- [description]
    """
    assert x.shape[0] == w.shape[1]
    matmul = np.dot(x,w)
    
    #output = s = f(X,W)
    output = function(matmul)
    
    return output

def backward_matrix_X(x, w, function=sigmoid):
    """[summary]
    
    Arguments:
        x {[type]} -- [description]
        w {[type]} -- [description]
        function {[type]} -- [description]
    """
    assert x.shape[0] == w.shape[1]
    
    ### matrix multiplication
    N = np.dot(x,w)
    
    ### feeding N to sigmoid s = f(x,y)
    output = sigmoid(N) ##forward pass done 
    
    ##backward 
    dSdN = derivative(sigmoid,output) #or derivative_sigmoid(output)
    
    ## w.r.t  X
    dNdX = np.transpose(w,(1,0))
    
    return dSdN * dNdX
    
    