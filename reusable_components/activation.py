import numpy as np 
from param_operation import ParamOperation

class Sigmoid(ParamOperation):
    
    def __init__(self):
        super().__init__()
        
    def _output(self):
        return (1/1+np.exp(-self.input_ndarray))
    
    def _input_grad(self, output_grad):
        
        sigmoid_backward = self.output(1- self.output)
        input_grad = sigmoid_backward * output_grad
        return input_grad
    
