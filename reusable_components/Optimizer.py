import numpy as np 

class Optimizer():
    '''
    Base class for optimizer
    '''
    def __init__(self,lr=0.01):
        self.lr = lr
        
    
    def step(self):
        '''
        Every optimizer will have "step" method
        '''
        pass    
    

class SGD(Optimizer):
    
    def __init__(self,lr=0.01):
        super().__init__(lr)
        
    def step(self):
        '''
        For each parameter, adjust in the appropriate direction, with the
        magnitude of the adjustment based on the learning rate.
        '''
        for (param, param_grad) in zip(self.net.params(),
                                       self.net.param_grads()):

            param -= self.lr * param_grad    