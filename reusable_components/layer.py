import numpy as np 
from base_class_nn import BaseClassNN
from param_operation import ParamOperation




class Layer(object):
    '''
    layer" of neurons in a neural network.
    '''

    def __init__(self, neurons):
     
        '''
        The number of "neurons" roughly corresponds to the "breadth" of the
        layer
        '''
        self.neurons = neurons
        self.first = True
        self.params = []
        self.param_grads = []
        self.operations = []

    def _setup_layer(self, num_in):
        '''
        The _setup_layer function must be implemented for each layer.
        '''
        raise NotImplementedError()

    def forward(self, input_):
        '''
        Passes input forward through a series of operations.
        '''
        if self.first:
            self._setup_layer(input_)
            self.first = False

        self.input_ = input_

        for operation in self.operations:

            input_ = operation.forward(input_)

        self.output = input_

        return self.output

    def backward(self, output_grad):
        '''
        Passes output_grad backward through a series of operations.
        Checks appropriate shapes.
        '''

        assert self.output.shape == output_grad.shape

        for operation in reversed(self.operations):
            output_grad = operation.backward(output_grad)

        input_grad = output_grad

        self._param_grads()

        return input_grad

    def _param_grads(self):
        '''
        Extracts the _param_grads from a layer's operations.
        '''

        self.param_grads = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.param_grads.append(operation.param_grad)

    def _params(self):
        '''
        Extracts the _params from a layer's operations.
        '''

        self.params = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.params.append(operation.param)