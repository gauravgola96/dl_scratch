import numpy as np 

class BaseClassNN():
    """
    Base class to perform some basic operations in NN 
    """
    
    def __init__(self):
        pass 
    
    def forward_pass(self, input_ndarray):
        
        """[forward pass calculation in NN ]
        Arguments:
            input_ndarray {[type]} -- [ndarray]
        """
        self.input_ndarray = input_ndarray
        
        self.output = self._output()
        
        return self.output
    
    def backward_pass(self, output_grad):
        """[summary]
        
        Arguments:
            output_grad {[type]} -- [ndarray, Gradients loss]
        """
        assert self.output.shape == output_grad.shape
        
        self.input_grad = self._input_grad(output_grad)
        
        assert self.input_grad.shape == self.input_grad.shape
        
        return self.input_grad


    def _output(self):
        '''
        The _output method must be defined for each Operation.
        '''
        raise NotImplementedError()


    def _input_grad(self, output_grad):
        '''
        The _input_grad method must be defined for each Operation.
        '''
        raise NotImplementedError()
        
        