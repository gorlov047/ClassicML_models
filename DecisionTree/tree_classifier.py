from __future__ import annotations
from typing import Tuple
import numpy as np
from decision_tree import DecisionTree
Num = np.number
Arr = np.ndarray


class DecisionTreeClassifier(DecisionTree):
    """A decision tree classifier.
    
    The function to measure the quality of a split - Gini.
    Gini is equal to the mathematical expectation of the number of
    incorrectly classified objects if we assign random labels to them
    from a discrete distribution given by probabilities in the terminal node.
        
    Parameters
    ----------
    max_depth : int, default=None
        The maximum depth of the tree.

    min_samples_split : int, default=None
        The minimum number of samples required to split an internal node:

    min_samples_leaf : int, default=None
        The minimum number of samples required to be at a leaf node.
    """

    def __init__(
        self,
        max_depth: int = None,
        min_samples_split: int = None,
        n_feats: str = None,
        seed: int = None
    )-> None:
        super().__init__(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            n_feats = n_feats,
            seed = seed
        )

    def _leaf_value(self, node: dict, y: Arr)-> Num:
        classes, counts = np.unique(y, return_counts=True)
        node["value"] = classes[counts.argmax()]
        probs = np.zeros(self.n_classes)
        probs[classes - 1] = counts / y.shape
        node["probs"] = probs

    def _find_best_split(
        self,
        feature_vector: Arr,
        target_vector: Arr
    )-> Tuple[Arr, Arr, Num, Num]:
        sorted_inds = np.argsort(feature_vector)
        target_vector = target_vector[sorted_inds]
        sorted_feature, uniq_sorted_inds = np.unique(feature_vector[sorted_inds],
                                                     return_index=True)
        tresholds = (sorted_feature[1:] + sorted_feature[:-1]) / 2
        classes, counts = np.unique(target_vector, return_counts=True)
        n_classes = classes.shape[0]
        left_freqs = np.tile(target_vector, (n_classes, 1))
        for i in range(n_classes):
            left_freqs[i] = np.where(left_freqs[i] == i, 1, 0)
        left_freqs = np.cumsum(left_freqs, axis=1)[:, uniq_sorted_inds - 1][:, 1:]
        right_freqs = (np.repeat(counts, target_vector.shape[0]).
                       reshape(n_classes, -1)[:, uniq_sorted_inds - 1][:, 1:] - left_freqs)
        left_n = left_freqs.sum(axis=0)
        right_n = target_vector.shape[0] - left_n
        left_freqs = left_freqs / (left_n + 1e-8)
        right_freqs = right_freqs / (right_n + 1e-8)
        gini = (left_n * np.sum(left_freqs ** 2, axis=0) +
                right_n * np.sum(right_freqs ** 2, axis=0))
        best_ind = np.argmax(gini)
        treshold_best = tresholds[best_ind]
        gini_best = gini[best_ind]
        return tresholds, gini, treshold_best, gini_best

    def predict_proba(self, X: Arr) -> Arr:
        """Predict class probabilities of the input samples X.

        The predicted class probability is the fraction of samples of the same
        class in a leaf.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        
        Returns
        -------
        y : array-like of shape (n_samples,)
            The predict probabilities.
        """
        return np.array([self._predict_node(x, self._tree)["probs"].tolist() for x in X])
