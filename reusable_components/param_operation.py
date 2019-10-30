from base_class_nn import BaseClassNN

class ParamOperation(BaseClassNN):
    
    def _init__(self, param):
        super().__init__()
        self.param = param
    
    def backward(self, output_grad):
        '''
        Calls self._input_grad and self._param_grad.
        Checks appropriate shapes.
        '''

        assert self.output.shape == output_grad.shape

        self.input_grad = self._input_grad(output_grad)
        self.param_grad = self._param_grad(output_grad)

        assert self.input_ndarray.shape == self.input_grad.shape
        assert self.param.shape == self.param_grad.shape

        return self.input_grad

    def _param_grad(self, output_grad):
        '''
        Every subclass of ParamOperation must implement _param_grad.
        '''
        raise NotImplementedError()
    