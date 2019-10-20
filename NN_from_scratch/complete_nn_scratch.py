#Reference complete_nn.png
import numpy as np 

def sigmoid(value):
    return(1/1+np.exp(-value))

def derivative_sigmoid(value):
    return sigmoid(value)*(1-sigmoid(value))


def forward_pass(X_ndarray, y_ndarray, W):
    """[summary]
    
    Arguments:
        X_ndarray {[type]} -- [description]
        y_ndarray {[type]} -- [description]
        W {[type]} -- [description]
    """
    M1 = np.dot(X_ndarray,W['W1'])
    
    #Adding bias
    N1 = M1 + W['B1']

    #Sigmoid layer
    O1 = sigmoid(N1)
    
    M2 = np.dot(O1,W['W2'])
    
    P = M2 + W['B2']
    
    loss = np.mean(np.power(p - y_ndarray,2))
    
    forward_info = {}
    forward_info['X'] = X_ndarray
    forward_info['Y'] = y_ndarray
    forward_info['P'] = P
    forward_info['M1'] = M1
    forward_info['M2'] = M2
    forward_info['N1'] = N1
    forward_info['O1'] = O1
    
    return loss, forward_info



def backpass(forward_info , W):
    """[calculate dLdW1 , dLdB1, dLdW2, dLdB2]
    
    Arguments:
        forward_info {[type]} -- [description]
        W {[type]} -- [description]
    """

    batch_size = forward_info['X'].shape[0]
    
    #L = (P - Y)^2
    dLdP = -2 * (forward_info['Y'] - forward_info['P'])
    
    dPdB2 = np.ones_like(W['B2'])
    ####
    dLdB2 = np.dot(dLdP,dPdB2)
    
    #M2 = np.dot(O1,W['W2'])
    dPdM2  = np.ones_like(forward_info['M2'])

    dM2dO1 = np.transpose(W['W2'],(1,0))
    
    dM2dW2 = np.transpose(forward_info['O1'],(1,0))    
    
    ##sigmoid 
    dO1dN1 = derivative_sigmoid(forward_info('N1'))
    
    dN1dM1 = np.ones_like(forward_info['M1'])
    dN1dB1 = np.ones_like(W['B1'])
    
    dM1dW1 = np.transpose(forward_info['X'],(1,0))
    
 
    dLdW2 = np.dot(dM2dW2, dLdP)
    dLdB2 = np.dot(dLdP, dPdB2)
    
    dLdM2 = dLdP * dPdM2
    dLdO1 = np.dot(dLdM2, dM2dO1)
    dLdN1 = dLdO1 * dO1dN1

    
       
    #dLdW1 = dLdP * dPdM2 * dM2dO1 * dO1dN1 * dN1dM1 * dM1dW1
    dLdM1 = dLdN1 * dN1dM1
    dLdW1 = np.dot(dM1dW1,dLdM1)
    
    # dLdB1 = dLdP * dPdM2 * dM2dO1 * dO1dN1 * dN1dB1
    dLdB1 = (dLdN1 * dN1dB1).sum(axis=0)

    loss_gradients  = {}

    loss_gradients['W2'] = dLdW2
    loss_gradients['B2'] = dLdB2.sum(axis=0)
    loss_gradients['W1'] = dLdW1
    loss_gradients['B1'] = dLdB1.sum(axis=0)
    
    return dLdW1, dLdB1, dLdW2, dLdB2
    
    
    
def predict(X_ndarray,W):        
    '''
    Generate predictions from the step-by-step neural network model.
    '''
    M1 = np.dot(X_ndarray, W['W1'])

    N1 = M1 + W['B1']

    O1 = sigmoid(N1)

    M2 = np.dot(O1, W['W2'])

    P = M2 + W['B2']

    return P