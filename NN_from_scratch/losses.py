import numpy as np


def mae(pred, y_ndarray):
    return np.mean(np.abs(pred - y_ndarray))


def mse(pred, y_ndarray):
    return np.mean(np.power(pred - y_ndarray, 2))


def rmse(pred, y_ndarray):
    return np.sqrt(np.mean(np.power(pred - y_ndarray, 2)))
