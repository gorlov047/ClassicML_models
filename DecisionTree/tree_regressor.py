from __future__ import annotations
from typing import Tuple
import numpy as np
from decision_tree import DecisionTree
Num = np.number
Arr = np.ndarray

class DecisionTreeRegressor(DecisionTree):
    """A decision tree regressor.
    
    The function to measure the quality of a split - MSE. MSE is equal to
    variance reduction as feature selection criterion and minimizes the
    loss using the mean of each terminal node.
    
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
        min_samples_split: int = None
    )-> None:
        super().__init__(
            max_depth=max_depth,
            min_samples_split=min_samples_split
        )

    def _leaf_value(self, node: dict, y: Arr)-> None:
        node["value"] = y.mean()

    def _find_best_split(
        self,
        feature_vector: Arr,
        target_vector: Arr
    )-> Tuple[Arr, Arr, Num, Num]:
        def variance(sum_, sum_of_squares, number):
            return sum_of_squares / number - (sum_ / number) ** 2

        sorted_inds = np.argsort(feature_vector)
        sorted_feature, inds = np.unique(feature_vector[sorted_inds],
                                         return_index=True)
        tresholds = (sorted_feature[1:] + sorted_feature[:-1]) / 2
        left_sum = np.cumsum(target_vector[sorted_inds])[inds - 1][1:]
        left_sq_sum = np.cumsum(target_vector[sorted_inds] ** 2)[inds - 1][1:]
        left_count = np.arange(1, feature_vector.shape[0], 1)[inds - 1][1:]
        left_var = variance(left_sum, left_sq_sum, left_count)

        right_count = feature_vector.shape[0] - left_count
        right_sum = target_vector.sum() - left_sum
        right_sq_sum = (target_vector ** 2).sum() - left_sq_sum
        right_var = variance(right_sum, right_sq_sum, right_count)

        info_gain = -((left_count * left_var + right_count * right_var)
                      / feature_vector.shape[0])
        best_ind = np.argmax(info_gain)
        return tresholds, info_gain, tresholds[best_ind], info_gain[best_ind]
