
from __future__ import annotations
import numpy as np
from DecisionTree import tree_regressor as tr
from random_forest import RandomForest
Num =np.number
Arr = np.ndarray


class RandomForestRegressor(RandomForest):
    """Random forest regressor.
    
    Parameters
    ----------
    n_trees : the number of decision trees in the ensemble.
    seed: seed for the random number generator.
    """
    def __init__(
        self,
        n_trees: int = None,
        seed: int = None
    )-> None:
        super().__init__(
            tree=tr.DecisionTreeRegressor,
            n_feats="n/3",
            n_trees=n_trees,
            seed=seed
        )

    def predict(self, X: Arr)-> Arr:
        preds = np.array([tree.predict(X) for tree in self.trees])
        return np.array([preds.mean() for x in preds.T])
