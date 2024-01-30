from __future__ import annotations
from typing import List
import numpy as np
from base_descent import BaseDescent
from gradient_descents import get_descent

class LinearRegression:
    """Linear regression class.
    
    Parameters
    -------
    descent_config: gradient descent config
    tolerance: stopping criterion for square of euclidean norm of weight difference
    max_iter: stopping criterion for iterations
    """

    def __init__(self, descent_config: dict,
                 tolerance: float = 1e-4,
                 max_iter: int = 300
                )->None:
        self.descent: BaseDescent = get_descent(descent_config)

        self.tolerance: float = tolerance
        self.max_iter: int = max_iter

        self.loss_history: List[float] = []

    def fit(self, x: np.ndarray, y: np.ndarray)-> LinearRegression:
        """Fitting descent weights for x and y dataset.
        
        Parameters
        -------
        x: features array
        y: targets array
        
        Returns
        -------
        fitted linear regressor
        """
        self.loss_history.append(self.calc_loss(x, y))
        for _ in range(self.max_iter):
            diff_w = self.descent.update_weights(self.descent.calc_gradient(x, y))
            self.loss_history.append(self.calc_loss(x, y))
            if np.dot(diff_w.T, diff_w) < self.tolerance:
                break
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """ Predicting targets for x dataset.
        
        Parameters
        -------
        x: features array
        
        Returns
        -------
        predictied values
        """
        return self.descent.predict(x)

    def calc_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculating loss for x and y dataset.
        
        Parameters
        -------
        x: features array
        y: targets array
        
        Returns
        -------
        loss function
        """
        return self.descent.calc_loss(x, y)
