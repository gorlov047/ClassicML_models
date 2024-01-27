from __future__ import annotations
from typing import Tuple
from abc import ABCMeta, abstractmethod
import numpy as np
Num =np.number
Arr = np.ndarray


class DecisionTree(metaclass=ABCMeta):
    """ Base class for decision trees.
    
    Warning: This class should not be used directly.
    Use derived classes instead.
    
    Parameters
    ----------
    max_depth : int, default=None
        The maximum depth of the tree.

    min_samples_split : int, default=None
        The minimum number of samples required to split an internal node:

    min_samples_leaf : int, default=None
        The minimum number of samples required to be at a leaf node.
    """

    @abstractmethod
    def __init__(
        self,
        max_depth: int = None,
        min_samples_split: int = None,
    )-> None:
        self._tree = {}
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    @abstractmethod
    def _find_best_split(
        self,
        feature_vector: Arr,
        target_vector: Arr
    )-> Tuple[Arr, Arr, Num, Num]:
        pass

    @abstractmethod
    def _leaf_value(self, node: dict, y: Arr)-> None:
        pass

    def _fit_node(
        self,
        sub_x: Arr,
        sub_y: Arr,
        node: dict,
        cur_depth: int = 0
    ) -> None:
        def set_terminal(node, sub_y):
            node["type"] = "terminal"
            self._leaf_value(node, sub_y)

        if (self.max_depth is not None and
            cur_depth == self.max_depth):
            set_terminal(node, sub_y)
            return

        if (self.min_samples_split is not None and
            sub_y.shape[0] < self.min_samples_split):
            set_terminal(node, sub_y)
            return

        if np.all(np.isclose(sub_y, sub_y[0])):
            set_terminal(node, sub_y)
            return

        feature_best, threshold_best, gain_best, split = None, None, None, None
        for feature in range(sub_x.shape[1]):
            feature_vector = sub_x[:, feature].astype(np.float64)

            if np.all(np.isclose(feature_vector, feature_vector[0])):
                continue

            _, _, threshold, gain = self._find_best_split(feature_vector, sub_y)
            if gain_best is None or gain > gain_best:
                feature_best = feature
                gain_best = gain
                split = feature_vector < threshold
                threshold_best = threshold

        if feature_best is None:
            set_terminal(node, sub_y)
            return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best
        node["threshold"] = threshold_best
        cur_depth += 1

        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_x[split], sub_y[split],
                       node["left_child"], cur_depth)
        self._fit_node(sub_x[np.logical_not(split)], sub_y[np.logical_not(split)],
                       node["right_child"], cur_depth)

    def _predict_node(self, sample: Arr, node: dict) -> float:
        if node["type"] == "terminal":
            return node

        feat = node["feature_split"]
        threshold = node["threshold"]

        if sample[feat] < threshold:
            return self._predict_node(sample, node["left_child"])
        return self._predict_node(sample, node["right_child"])

    def fit(self, X: Arr, y: Arr) -> DecisionTree:
        """Build a decision tree from the training set (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.  

        y : array-like of shape (n_samples,)

        Returns
        -------
        self : DecisionTree
            Fitted estimator.

        """

        self._fit_node(X, y, self._tree)
        return self

    def predict(self, X: Arr) -> Arr:
        """Predict class or regression value for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        
        Returns
        -------
        y : array-like of shape (n_samples,)
            The predict values.
        """
        return np.array([self._predict_node(x, self._tree)["value"] for x in X])
