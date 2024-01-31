import numpy as np
from linear_regression import LinearRegression
from loss_functions import CrossEntropy

class LogisticRegression(LinearRegression):
    """Logistic regression class.
    Warning:  use only CrossEntropy loss in descent_config
    The loss minimised is the multinomial loss fit across
    the entire probability distribution,even when the data is binary.
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
        super().__init__(descent_config, tolerance, max_iter)
        if self.descent.loss_function.__class__ is not CrossEntropy:
            raise ValueError

    def predict_proba(self, x: np.ndarray)-> np.ndarray:
        """ Predicting probabilities for x dataset.

        Parameters
        -------
        x: features array

        Returns
        -------
        predictied probabilities for each class
        """
        y_pred = self.descent.predict(x)
        probs = np.exp(y_pred)
        probs /= np.expand_dims(probs.sum(axis=1), 1)
        return probs

    def predict(self, x: np.ndarray)-> np.ndarray:
        return self.predict_proba(x).argmax(axis=1)
