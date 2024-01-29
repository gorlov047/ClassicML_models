from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple
import numpy as np
from loss_functions import LossFunction, MSE

@dataclass
class LearningRate:
    """LeariningRate dataclass.
    Changes the learning rate depending on the step of the gradient descent.
    """
    lambda_: float = 1e-3
    s0: float = 1
    p: float = 0.5

    iteration: int = 0

    def __call__(self)->np.float:
        """
        Calculate learning rate according to lambda (s0/(s0 + t))^p formula.
        """
        self.iteration += 1
        return self.lambda_ * (self.s0 / (self.s0 + self.iteration)) ** self.p

class BaseDescent(ABC):
    """A base class for gradient decent.
    
    Warning: This class should not be used directly.
    Use derived classes instead.
    
    Parameters
    ----------
    dimension: feature space dimension
    lambda_: learning rate parameter
    loss_function: optimized loss function
    """

    @abstractmethod
    def __init__(self, dimension: Tuple[int, int], lambda_: float = 1e-3,
                 loss_function: LossFunction = MSE()
                )->None:
        self.w: np.ndarray = np.random.rand(*dimension)
        self.lr: LearningRate = LearningRate(lambda_=lambda_)
        self.loss_function: LossFunction = loss_function

    @abstractmethod
    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """Update weights with respect to gradient.
        
        Parameters
        -------
        gradient: gradient of loss
        
        Returns
        -------
        weight difference (w_{k + 1} - w_k)
        """

    @abstractmethod
    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculate gradient of loss function with respect to weights
        
        Parameters
        -------
        x: features array
        y: targets array
        Returns
        -------
        gradient
        """

    def calc_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate loss for x and y with our weights
         
        Parameters
        -------
        x: features array
        y: targets array
        Returns
        -------
        the value of the loss function 
        """
        return self.loss_function.calc_loss(y, np.dot(x, self.w))

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Calculate predictions for x
        
        Parameters
        -------
        x: features array
        
        Returns
        -------
        predicted values
        """
        return np.dot(x, self.w)

class BaseDescentReg(BaseDescent, ABC):
    """A base class for gradient descne with regularization.
    
    Parameters
    -------
    mu: regularization coefficient
    type_: 'l2': add a L2 penalty term;
          'l1': add a L1 penalty term;
    """

    @abstractmethod
    def __init__(self, *args, mu: float = 0, type_: str = 'l2', **kwargs)->None:
        super().__init__(*args, **kwargs)

        self.mu = mu
        self.type = type_

    @abstractmethod
    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:

        weights_norm = np.copy(self.w)
        weights_norm[-1] = 0
        if self.type_ == 'l2':
            weights_norm = weights_norm ** 2

        return super().calc_gradient(x, y) + sum(abs(weights_norm)) * self.mu / 2
