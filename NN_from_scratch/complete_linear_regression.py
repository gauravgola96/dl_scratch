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


# Refer backpass.png
def loss_gradients(forward_info, W):
    """Calculate dLdW and dLdB
    
    Arguments:
        forward_info {[type]} -- [description]
        W {[type]} -- [description]
    """
    batch_size = forward_info['X'].shape[0]
    
    #L = np.power(Y - P,2)
    dLdP = -2 * (forward_info['y'] - forward_info['P'])
    
    #P = np.dot(X,W) + B = X.W + B = N + B 
    #dPbN = B #constant
    #N = np.dot(X,W)
    dPdN = np.ones_like(forward_info['N'])
    
    dPdB = np.ones_like(W['B'])
    
    #N = np.dot(x,W)
    dNdW = np.transpose(forward_info['X'], (1,0))
    
    #NOW calculate dLdW
    dLdN = dLdP * dPdN
    ###################
    dLdW = np.dot(dNdW , dLdN)
    
    #Now calculate dLdB
    dLdB = np.dot(dLdP ,dPdB)
    
    loss_gradients = {}
    loss_gradients['W'] = dLdW
    loss_gradients['B'] = dLdB
    
    return loss_gradients


# Basic intution for using learning rate in dict
#this is how we can update Weights
#for key in weights.keys():
#    weights[key] -= learning_rate * loss_grads[key]