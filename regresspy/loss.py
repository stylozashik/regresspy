import numpy as np
from numpy import ndarray
from math import sqrt

def mae(pred: ndarray, label: ndarray) -> ndarray:
    return np.sum(np.abs(pred - label)) / np.shape(pred)


def sse(pred: ndarray, label: ndarray) -> ndarray:
    return np.square(np.sum((pred - label) / np.shape(pred)))


def mse(pred: ndarray, label: ndarray) -> ndarray:
    return np.square(pred - label).mean()


def rmse(pred: ndarray, label: ndarray) -> float:
    return sqrt(np.square(pred - label).mean())
