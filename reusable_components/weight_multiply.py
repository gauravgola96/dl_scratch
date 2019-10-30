import numpy as np 
from param_operation import ParamOperation


class WeightMultiply(ParamOperation):
    '''
    Weight multiplication operation for a neural network.
    '''

    def __init__(self, W):
        '''
        Initialize Operation with self.param = W.
        '''
        super().__init__(W)

    def _output(self):
        '''
        Compute output.
        '''
        return np.dot(self.input_ndarray, self.param)

    def _input_grad(self, output_grad):
        '''
        Compute input gradient.
        '''
        return np.dot(output_grad, np.transpose(self.param, (1, 0)))

    def _param_grad(self, output_grad):
        '''
        Compute parameter gradient.
        '''
        return np.dot(np.transpose(self.input_ndarray, (1, 0)), output_grad)



class BiasAdd(ParamOperation):
    '''
    Compute bias addition.
    '''

    def __init__(self,B):
        '''
        Initialize Operation with self.param = B.
        Check appropriate shape.
        '''
        assert B.shape[0] == 1

        super().__init__(B)


    def _output(self):
        '''
        Compute output.
        '''
        return self.input_ndarray + self.param


    def _input_grad(self, output_grad):
        '''
        Compute input gradient.
        '''
        return np.ones_like(self.input_ndarray) * output_grad


    def _param_grad(self, output_grad):
        '''
        Compute parameter gradient.
        '''
        param_grad = np.ones_like(self.param) * output_grad
        return np.sum(param_grad, axis=0).reshape(1, param_grad.shape[1])
