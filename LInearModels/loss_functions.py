from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

class LossFunction(ABC):
    """ Base class for loss function.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """
    @abstractmethod
    def calc_gradient(self, x: np.ndarray, y: np.ndarray, w: np.ndarray)-> np.ndarray:
        """Calculate gradient of loss function with respect to weights.
 
        Parameters
        ----------
        x: features array
        y: targets array
        w: weights array

        Returns
        -------
        y : gradient
        """

    @abstractmethod
    def calc_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate loss based on real and predicted target.

        Parameters
        ----------
        y_true: real targets
        y_pred: predicted targets
        w: weights array

        Returns
        -------
        y: loss
        """

class MSE(LossFunction):
    """Mean squared error loss.
    """
    def calc_gradient(self, x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        return -2 / x.shape[0] * np.dot(x.T, y - np.dot(x, w))

    def calc_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return ((y_pred - y_true) ** 2).mean()


class MAE(LossFunction):
    """Mean absolute error loss.
    """
    def calc_gradient(self, x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        return -1 / x.shape[0] * np.dot(x.T, np.sign(y - np.dot(x, w)))

    def calc_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.abs(y_true - y_pred).mean()


class LogCosh(LossFunction):
    """LogCosh loss.
    """
    def calc_gradient(self, x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        return 1 / x.shape[0] * np.dot(x.T, np.tanh(np.dot(x, self.w) - y))

    def calc_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.log(np.cosh(y_true - y_pred)).mean()

class Huber(LossFunction):
    """Huber loss.

    This function is quadratic for small values of the
    delta modulus, and linear for large values.

    Parameters
    ----------
    delta: for values modulo large, the function is linear
    """
    def __init__(self, delta: float = 1.35)->None:
        self.delta = delta

    def calc_gradient(self, x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        y_pred = np.dot(x, w)
        diff = np.abs(y_pred - y)
        return np.dot(x.T,
                      y - np.where(diff <= self.delta, y_pred, np.sign(y - y_pred))
                     ) / x.shape[0]

    def calc_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        diff = np.abs(y_pred - y_true)
        return np.where(diff <= self.delta, diff ** 2 / 2,
                        self.delta * (diff - self.delta / 2)
                       ).mean()


class CrossEntropy(LossFunction):
    """CrossEntropy Loss.
    When there are only two classes, this loss function
    is equivalent to the LogLoss loss function.
    
    Parameters
    -------
    n_classes: number of classes
    """
    def __init__(self, n_classes: int = 2)->None:
        self.n_classes = n_classes

    def calc_gradient(self, x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        def get_one_hot(targets, nb_classes):
            res = np.eye(nb_classes, dtype=int)[np.array(targets).reshape(-1)]
            return res.reshape(list(targets.shape)+[nb_classes])

        y_pred = np.dot(x, w)
        probs = np.exp(y_pred)
        probs /= np.expand_dims(probs.sum(axis=1), 1)
        return np.dot(x.T, probs - get_one_hot(y, self.n_classes))

    def calc_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        probs = np.exp(y_pred)
        probs /= np.expand_dims(probs.sum(axis=1), 1)
        return -np.log(probs[np.arange(y_true.shape[0]), y_true]).mean()
