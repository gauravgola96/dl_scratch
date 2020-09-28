import random
import numpy as np
import math


##
def sigmoid(x):
    return (1.0 / (1 + np.exp(-x)))


def sigmoid_derivation(values):
    dervative = values(1 - values)
    return dervative


def tanh(x):
    return np.tanh(x)


def tanh_derivative(values):
    derivative = 1 - values ** 2
    return derivative


# probabilty distribution of uniform distribution is 1/(b-a)
# interval [a.b)

def rand_arr(a, b, *args):
    np.random.seed(1)
    return np.random.rand(*args) * (b - a) + a


H_size = 100  # Size of the hidden layer
T_steps = 25  # Number of time steps (length of the sequence) used for training
learning_rate = 1e-1  # Learning rate
weight_sd = 0.1  # Standard deviation of weights for initialization
z_size = H_size + X_size  # Size of concatenate(H, X) vector


class Param:
    def __init__(self, name, value):
        self.name = name
        self.v = value
        self.d = np.zeros_like(value)  # derivative
        self.m = np.zeros_like(value)  # momentum


class Parameters:

    def __init__(self):
        self.W_f = Param('W_f', np.random.rand(H_size, z_size) * weight_sd + 0.5)

        self.b_f = Param('b_f', np.zeros((H_size, 1)))

        self.W_i = Param('W_i', np.random.randn(H_size, z_size) * weight_sd + 0.5)

        self.b_i = Param('b_i', np.zeros((H_size, 1)))

        self.W_c = Param('W_c', np.random.rand(H_size, z_size) * weight_sd)

        self.b_c = Param('b_c', np.zeros((H_size, 1)))

        self.W_o = Param('W_o', np.random.rand(H_size, z_size) * weight_sd + 0.5)

        self.b_o = Param('b_o', np.zeros((H_size, 1)))

        # for final layer

        self.W_v = Param('W_v',
                         np.random.randn(X_size, H_size) * weight_sd)
        self.b_v = Param('b_v',
                         np.zeros((X_size, 1)))

    def all_layers(self):
        return [self.W_f, self.W_i, self.W_C, self.W_o, self.W_v,
                self.b_f, self.b_i, self.b_c, self.b_o, self.b_v]
