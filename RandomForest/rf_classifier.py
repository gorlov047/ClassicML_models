from __future__ import annotations
import numpy as np
from DecisionTree import tree_classifier as tc
from random_forest import RandomForest
Num =np.number
Arr = np.ndarray


class RandomForestClassifier(RandomForest):
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
        tree=tc.DecisionTreeClassifier,
        n_feats="sqrt",
        n_trees=n_trees,
        seed=seed
        )

    def predict(self, X: Arr)-> Arr:
        preds = np.array([tree.predict(X) for tree in self.trees])
        return np.array([np.bincount(x).argmax() for x in preds.T])
