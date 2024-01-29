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
        iteration = 0
        sq_norm_w_diff = self.tolerance

        while iteration < self.max_iter and sq_norm_w_diff >= self.tolerance:
            iteration += 1
            self.loss_history.append(self.calc_loss(x, y))
            old_w = np.copy(self.descent.w)
            self.descent.update_weights(self.descent.calc_gradient(x, y))
            diff_w = old_w - self.descent.w
            if np.isnan(diff_w).sum():
                break
            sq_norm_w_diff = np.dot(diff_w, diff_w)

        self.loss_history.append(self.calc_loss(x, y))
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
