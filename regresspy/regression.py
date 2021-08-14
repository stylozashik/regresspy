from typing import Dict, Tuple

import numpy as np
from numpy import ndarray

from loss import mae, sse, mse, rmse
from gradient_descent import forward, backward
from regresspy import loss


class Regression(object):
    def __init__(self, epochs=50, learning_rate=0.001):
        self._epochs = epochs
        self._lr = learning_rate
        self._weights = {}
        self._X = None
        self._Y = None

    def fit(self, X: ndarray, Y: ndarray) -> None:
        """Fits the data using gradient descent algorithm.
        Args:
            X (ndarray): Should be of shape (observations x features)
            Y (ndarray): Should be of shape (observations x 1)
        """
        assert X.shape[0] == Y.shape[0]

        self._initialize_weights(X.shape)
        assert self._weights['W'].shape == (X.shape[1], 1)
        assert self._weights['B'].shape == (1, 1)

        self._train(X, Y)

    def predict(self, X: ndarray) -> ndarray:
        """Predicts new data with the fitted weights and bias.
        Args:
            X (ndarray): new dataset of shape (observations x features)
        Returns
            (ndarray): predictions of shape (observations x 1)
        """
        predictions =  self._weights['W']*X + self._weights['B']
        return predictions

    def score(self, X: ndarray, Y: ndarray, metric='rmse') -> float:
        """Returns the score of the fitted data. Possible metrics include
        'mae', 'sse', 'mse', and 'rmse'.
        Args:
            X (ndarray): dataset of shape (observations x features)
            Y (ndarray): labels of shape (observations x 1)
            metric (str): 'mae', 'sse', 'mse', or 'rmse'
        Returns:
            (float): score of the fitted line.
        """
        metrics = {
            'mae': mae,
            'sse': sse,
            'mse': mse,
            'rmse': rmse
        }

        predictions =  self._weights['W']*X + self._weights['B']
        if metric == 'mae':
            score = mae(Y , predictions)
        elif metric == 'sse':
            score = sse(Y , predictions)
        elif metric == 'mse':
            score = mse(Y , predictions)
        else:
            score = rmse(Y , predictions)

        return score

    def _initialize_weights(self, shape: Tuple[int, int]) -> None:
        """The shape of the weights will be (features x 1), and the shape
        of the bias will be (1,1).
        """
        self._weights = {
            'W': np.random(X.shape[1]),
            'B': np.random(1, 1)
        }

    def _train(self, X: ndarray, Y: ndarray) -> None:
        """Train data using gradient descent
        """
        for i in range(self._epochs):
            print('Epoch: ', i + 1)
            loss, info =  forward(X , Y , self._weights['W'])
            print('Loss: ', loss)
            grads =  backward(info , self._weights['W'])
            self._weights['W'] = self._weights['W'] -  loss
            self._weights['B'] = self._weights['B'] -  loss
