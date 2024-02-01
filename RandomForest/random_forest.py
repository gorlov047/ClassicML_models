from __future__ import annotations
from abc import ABCMeta, abstractmethod
import numpy as np
from DecisionTree import decision_tree as dt
Num =np.number
Arr = np.ndarray


class RandomForest(metaclass=ABCMeta):
    """Base class for random forest
    
    Warning: This class should not be used directly.
    Use derived classes instead.
    
    An ensemble of decision trees where each tree is trained on
    a bootstrap sample and each split is calculated sing a
    random subset of the features in the input.

    Parameters
    ----------
    tree: base estimator.
    n_feats: specifies the number of features to sample on each split.
        "sqrt" - heuristics for classification;
        "n/3" - heuristics for regression.
    n_trees : the number of decision trees in the ensemble.
    seed: seed for the random number generator.
    """

    @abstractmethod
    def __init__(
        self,
        tree: dt.DecisionTree = None,
        n_feats: str = None,
        n_trees: int = None,
        seed: int = None
    )-> None:
        self.tree = tree
        self.n_trees = n_trees
        self.n_feats = n_feats
        self.seed=seed
        if seed is not None:
            np.random.seed(seed)
        self.trees = []

    def _bootstrap_sample(self, X: Arr, y: Arr)-> Arr:
        n = X.shape[0]
        idxs = np.random.choice(n, n, replace=True)
        return X[idxs], y[idxs]

    def fit(self, X: Arr, y: Arr)-> RandomForest:
        """Fit n_trees of decision trees each on bootstrap samples.
        
        Parameters
        ----------
        X: features array
        y: targets array
        
        Returns
        -------
        fitted random forest
        """
        for _ in range(self.n_trees):
            X_samp, y_samp = self._bootstrap_sample(X, y)
            tree = self.tree(n_feats=self.n_feats,
                             seed=self.seed)
            self.trees.append(tree.fit(X_samp, y_samp))

    @abstractmethod
    def predict(self, X: Arr)-> Arr:
        """Predict class or regression value for X.

        Parameters
        ----------
         X: features array
        
        Returns
        -------
        predictied values
        """
