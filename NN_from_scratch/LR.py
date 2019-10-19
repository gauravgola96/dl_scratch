'''
Linear regression
ref: http://www.stat.yale.edu/Courses/1997-98/101/linreg.htm


Equation : y = Bo + B1X1 + B2X2 + B3X3 + ...
Each xi being multiplied by another element wi and then the results being added together
Something we have done in foundation module code


'''
import numpy as np 

#Breaking down equation into nested function : loss(v(B,W),Y) 

def linear_reg(X_ndarray , Y_ndarray, W):
    """[Forward linear regression]
    
    Arguments:
        X_ndarray {[type]} -- [description]
        Y_ndarray {[type]} -- [description]
        W: weights {[Dict]} -- [Dict[str, ndarray]]
    """

    # assert batch sizes of X and y are equal
    assert X_ndarray.shape[0] == Y_ndarray.shape[0]

    # assert that matrix multiplication can work
    assert X_ndarray.shape[1] == W['W'].shape[0]

    # assert that B is simply a 1x1 ndarray as it bias
    assert W['B'].shape[0] == W['B'].shape[1] == 1

    # compute the operations on the forward pass
    N = np.dot(X_ndarray, W['W'])

    P = N + W['B']

    loss = np.mean(np.power(Y_ndarray - P, 2)) #MSE

    # save the information computed on the forward pass
    forward_info= W[str, Y_ndarray] = {}
    forward_info['X'] = X_ndarray
    forward_info['N'] = N
    forward_info['P'] = P
    forward_info['y'] = Y_ndarray

    return loss, forward_info

def derivative_mse_loss(Y,P):
    return -2 * (Y-P)




