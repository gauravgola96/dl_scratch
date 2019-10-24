import numpy as np 

class Loss(object):
    '''
    The "loss" of a neural network.
    '''

    def __init__(self):
        pass

    def forward(self, prediction, target):
        '''
        Computes the actual loss value.
        '''
        assert prediction.shape == target.shape

        self.prediction = prediction
        self.target = target

        loss_value = self._output()

        return loss_value

    def backward(self):
        '''
        Computes gradient of the loss value with respect to the input to the
        loss function.
        '''
        self.input_grad = self._input_grad()

        assert self.prediction.shape == self.input_grad.shape

        return self.input_grad

    def _output(self):
        '''
        Every subclass of "Loss" must implement the _output function.
        '''
        raise NotImplementedError()

    def _input_grad(self):
        '''
        Every subclass of "Loss" must implement the _input_grad function.
        '''
        raise NotImplementedError()
    
    
    
class MeanSquaredError(Loss):
    
    def __init__(self):
        
        super().__init__()

    def _output(self):
        '''
        Computes the per-observation squared error loss.
        '''
        loss = np.sum(np.power(self.prediction - self.target, 2)) / self.prediction.shape[0]

        return loss

    def _input_grad(self):
        '''
        Computes the loss gradient with respect to the input for MSE loss.
        '''

        return 2.0 * (self.prediction - self.target) / self.prediction.shape[0]