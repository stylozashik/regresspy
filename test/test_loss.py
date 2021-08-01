from regresspy.loss import mae,sse
import numpy as np


def test_mae():
    dummy_predictions = np.array([2.0, 4.0, 7.0, 8.0])
    dummy_labels = np.array([1.0, 3.0, 6.0, 7.0])
    dummy_mae = 1.0
    assert mae(dummy_predictions, dummy_labels) == dummy_mae


def test_sse():
    dummy_predictions = np.array([3.0, 5.0, 8.0, 9.0])
    dummy_labels = np.array([1.0, 3.0, 6.0, 7.0])
    dummy_sse = 4.0
    assert sse(dummy_predictions, dummy_labels) == dummy_sse


def test_mse():
    pass


def test_rmse():
    pass
