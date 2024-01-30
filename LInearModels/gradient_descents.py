from typing import Dict, Tuple, Type
import numpy as np
from base_descent import BaseDescent, BaseDescentReg
from loss_functions import LossFunction, MSE

class VanillaGradientDescent(BaseDescent):
    """Full gradient descent class.
    """
    def __init__(self, dimension: Tuple[int, int], lambda_: float = 1e-3,
                loss_function: LossFunction = MSE()
            )->None:
        super().__init__(dimension, lambda_, loss_function)

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        step = -self.lr() * gradient
        self.w += step
        return step
    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.loss_function.calc_gradient(x, y, self.w)

class StochasticDescent(VanillaGradientDescent):
    """Stochastic gradient descent class.
    
    Parameters
    -------
    dimension: feature space dimension
    lambda_: learning rate parameter
    loss_function: optimized loss function
    batch_size: batch size for gradient estimation
    """
    def __init__(self, dimension: int, lambda_: float = 1e-3,
                 batch_size: int = 50, loss_function: LossFunction = MSE()
                )->None:
        super().__init__(dimension, lambda_, loss_function)
        self.batch_size = batch_size

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        indexes = np.random.randint(0, x.shape[0], self.batch_size)
        return self.loss_function.calc_gradient(x[indexes], y[indexes], self.w)

class MomentumDescent(VanillaGradientDescent):
    """Momentum gradient descent class.
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3,
                 loss_function: LossFunction = MSE()
                )->None:
        super().__init__(dimension, lambda_, loss_function)
        self.alpha: float = 0.9
        self.h: np.ndarray = np.zeros(dimension)

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        self.h = self.alpha * self.h + self.lr() * gradient
        self.w -= self.h
        return -self.h

class Adam(VanillaGradientDescent):
    """Adaptive Moment Estimation gradient descent class.
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3,
                 loss_function: LossFunction = MSE()
                )->None:
        super().__init__(dimension, lambda_, loss_function)
        self.eps: float = 1e-8

        self.m: np.ndarray = np.zeros(dimension)
        self.v: np.ndarray = np.zeros(dimension)

        self.beta_1: float = 0.9
        self.beta_2: float = 0.999

        self.iteration: int = 0

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * gradient
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * gradient ** 2
        self.iteration += 1
        step = (-self.lr() *  self.m / (1 - self.beta_1 ** self.iteration) /
                ((self.v / (1 - self.beta_2 ** self.iteration)) ** 0.5 + self.eps))
        self.w += step
        return step

class VanillaGradientDescentReg(BaseDescentReg, VanillaGradientDescent):
    """Full gradient descent with regularization class.
    """


class StochasticDescentReg(BaseDescentReg, StochasticDescent):
    """Stochastic gradient descent with regularization class.
    """


class MomentumDescentReg(BaseDescentReg, MomentumDescent):
    """Momentum gradient descent with regularization class.
    """


class AdamReg(BaseDescentReg, Adam):
    """Adaptive gradient algorithm with regularization class.
    """

def get_descent(descent_config: dict) -> BaseDescent:
    """Get descent by config.

    Parameters
    -------
    descent_config: dict of parameters
    Returns
    -------
    gradient descent with parameters from config
    """
    descent_name = descent_config.get('descent_name', 'full')
    regularized = descent_config.get('regularized', False)

    descent_mapping: Dict[str, Type[BaseDescent]] = {
        'full': VanillaGradientDescent if not regularized else VanillaGradientDescentReg,
        'stochastic': StochasticDescent if not regularized else StochasticDescentReg,
        'momentum': MomentumDescent if not regularized else MomentumDescentReg,
        'adam': Adam if not regularized else AdamReg
    }

    if descent_name not in descent_mapping:
        raise ValueError(f'Incorrect descent name, use one of these: {descent_mapping.keys()}')

    descent_class = descent_mapping[descent_name]

    return descent_class(**descent_config.get('kwargs', {}))
