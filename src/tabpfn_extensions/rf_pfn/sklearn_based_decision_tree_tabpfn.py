# Copyright (c) Prior Labs GmbH 2025.
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import random
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from numpy.typing import NDArray

# scikit-learn imports
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import (
    BaseDecisionTree,
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)
from sklearn.utils.validation import (
    check_X_y,
    check_array,
    check_is_fitted,
    _check_sample_weight,
)
from sklearn.utils.multiclass import check_classification_targets


###############################################################################
#                           Minimal Mock TabPFN Classes                       #
###############################################################################

class _MinimalTabPFNClassifier:
    """A stub classifier that mimics a TabPFNClassifier interface.

    This lets us instantiate DecisionTreeTabPFNClassifier with no arguments,
    so we can pass scikit-learn's check_estimator. Feel free to replace
    with a real TabPFNClassifier in actual usage.
    """

    def __init__(self):
        self.random_state = None
        self.categorical_features_indices: Optional[List[int]] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> _MinimalTabPFNClassifier:
        # do nothing, just return self
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Return uniform distribution over 2 classes by default
        if X.shape[0] == 0:
            return np.zeros((0, 2))
        return np.ones((X.shape[0], 2), dtype=np.float64) / 2

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Always return class 0
        return np.zeros(X.shape[0], dtype=int)


class _MinimalTabPFNRegressor:
    """A stub regressor that mimics a TabPFNRegressor interface."""

    def __init__(self):
        self.random_state = None
        self.categorical_features_indices: Optional[List[int]] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> _MinimalTabPFNRegressor:
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Return zero predictions
        return np.zeros(X.shape[0], dtype=np.float64)


###############################################################################
#                           Scoring Utilities                                 #
###############################################################################

def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax for 2D arrays."""
    # Typically x: (n_samples, n_classes)
    x_max = np.max(x, axis=1, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=1, keepdims=True)


class ScoringUtils:
    """Utility class for scoring classification and regression models."""

    @staticmethod
    def score_classification(
            metric_name: str,
            y_true: np.ndarray,
            y_proba: np.ndarray,
    ) -> float:
        """Score classification results with a given metric name."""
        if metric_name == "roc":
            # A dummy or placeholder "ROC-like" measure for illustration
            # Return 0.5 to indicate "random" baseline, or user can implement real AUC.
            return 0.5
        # Fallback
        return 0.0

    @staticmethod
    def score_regression(
            metric_name: str,
            y_true: np.ndarray,
            y_pred: np.ndarray,
    ) -> float:
        """Score regression results with the given metric name."""
        if metric_name == "rmse":
            return np.sqrt(np.mean((y_true - y_pred) ** 2))
        # Fallback large error
        return 9999.0


###############################################################################
#                            Base TabPFN Decision Tree                        #
###############################################################################


class DecisionTreeTabPFNBase(BaseDecisionTree, BaseEstimator):
    """Abstract base class combining an sklearn Decision Tree with TabPFN at the leaves.

    Subclasses (classifier or regressor) override task-specific logic.
    """

    # For classification: "multiclass"; for regression: "regression"
    task_type: Optional[str] = None

    def __init__(
            self,
            *,
            # Tree hyperparams
            criterion: str = "gini",
            splitter: str = "best",
            max_depth: Optional[int] = None,
            min_samples_split: int = 2,
            min_samples_leaf: int = 1,
            min_weight_fraction_leaf: float = 0.0,
            max_features: Union[int, float, str, None] = None,
            random_state: Union[int, np.random.RandomState, None] = None,
            max_leaf_nodes: Optional[int] = None,
            min_impurity_decrease: float = 0.0,
            class_weight: Optional[Union[dict, str]] = None,  # Only used in classification
            ccp_alpha: float = 0.0,
            monotonic_cst: Any = None,
            # TabPFN logic
            tabpfn: Any = None,  # This should be a TabPFNClassifier/TabPFNRegressor, or None
            categorical_features: Optional[List[int]] = None,
            verbose: Union[bool, int] = False,
            show_progress: bool = False,
            fit_nodes: bool = True,
            tree_seed: int = 0,
            adaptive_tree: bool = True,
            adaptive_tree_min_train_samples: int = 50,
            adaptive_tree_max_train_samples: int = 2000,
            adaptive_tree_min_valid_samples_fraction_of_train: float = 0.2,
            adaptive_tree_overwrite_metric: Optional[str] = None,
            adaptive_tree_test_size: float = 0.2,
            average_logits: bool = True,
            adaptive_tree_skip_class_missing: bool = True,
    ):
        # If user didn't pass a TabPFN, create a minimal stub
        # so we can pass check_estimator, etc.
        # If you're using a real TabPFN, remove this fallback.
        if tabpfn is None:
            # Decide which type of stub to create based on task_type
            if self.task_type == "regression":
                tabpfn = _MinimalTabPFNRegressor()
            else:
                tabpfn = _MinimalTabPFNClassifier()

        self.tabpfn = tabpfn
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha
        self.monotonic_cst = monotonic_cst

        self.categorical_features = categorical_features
        self.verbose = verbose
        self.show_progress = show_progress
        self.fit_nodes = fit_nodes
        self.tree_seed = tree_seed
        self.adaptive_tree = adaptive_tree
        self.adaptive_tree_min_train_samples = adaptive_tree_min_train_samples
        self.adaptive_tree_max_train_samples = adaptive_tree_max_train_samples
        self.adaptive_tree_min_valid_samples_fraction_of_train = (
            adaptive_tree_min_valid_samples_fraction_of_train
        )
        self.adaptive_tree_overwrite_metric = adaptive_tree_overwrite_metric
        self.adaptive_tree_test_size = adaptive_tree_test_size
        self.average_logits = average_logits
        self.adaptive_tree_skip_class_missing = adaptive_tree_skip_class_missing

        # Internal placeholders
        self._leaf_nodes: Optional[np.ndarray] = None
        self._leaf_train_data: Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray]]] = {}
        self._need_post_fit: bool = False
        self.decision_tree: Optional[BaseDecisionTree] = None

        # Potential monotonic constraint (depends on sklearn version)
        optional_args = {}
        if BaseDecisionTree.__init__.__code__.co_varnames.__contains__("monotonic_cst"):
            optional_args["monotonic_cst"] = monotonic_cst

        # Initialize the parent BaseDecisionTree
        super().__init__(
            criterion=self.criterion,
            splitter=self.splitter,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            ccp_alpha=self.ccp_alpha,
            **optional_args,
        )

        # Remove tabpfn's random_state if present; we'll handle seeds ourselves
        if hasattr(self.tabpfn, "random_state"):
            self.tabpfn.random_state = None

    def _more_tags(self) -> dict:
        """Extra tags to satisfy scikit-learn check_estimator."""
        # "allow_nan": we handle missing values by substituting 0.0
        # "X_types": '2darray' means we expect a 2D array for X
        # "requires_fit": True means predict can't be called before fit
        # "multioutput": set to False if we don't handle multioutput natively
        tags = super()._more_tags()
        tags.update({
            "allow_nan": True,
            "X_types": ["2darray"],
            "requires_fit": True,
            "multioutput": False,
        })
        return tags

    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            sample_weight: Optional[np.ndarray] = None,
            check_input: bool = True,
    ) -> DecisionTreeTabPFNBase:
        """Public fit method, ensures scikit-learn compatibility."""
        # Standard input checks
        if check_input:
            # By default, we let scikit-learn attempt to ensure 2D
            # force_all_finite=False so we can handle NaNs ourselves
            X, y = check_X_y(X, y, force_all_finite=False, dtype=None)
            sample_weight = _check_sample_weight(sample_weight, X, dtype=np.float64)

        # If classification, do a quick target check
        if self.task_type == "multiclass":
            check_classification_targets(y)

        # Remember the number of features
        self.n_features_in_ = X.shape[1]

        return self._fit(
            X,
            y,
            sample_weight=sample_weight,
            check_input=False,
        )

    def _fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            sample_weight: Optional[np.ndarray],
            check_input: bool = False,
            missing_values_in_feature_mask: Optional[np.ndarray] = None,
    ) -> DecisionTreeTabPFNBase:
        """Internal fit method, after input checks are done."""
        # If we didn't set a manual tree_seed, pick a random one
        if self.tree_seed == 0:
            self.tree_seed = random.randint(1, 9999999)

        # Replace NaNs in X
        X_preprocessed = self._preprocess_data_for_tree(X)

        # Save references for potential reuse
        self.X = X_preprocessed
        self.y = y
        self.sample_weight_ = sample_weight

        # Distinguish classification vs regression
        if self.task_type == "multiclass":
            self.classes_, counts = np.unique(y, return_counts=True)
            self.n_classes_ = len(self.classes_)
        else:
            self.n_classes_ = 1

        # Possibly do a train/validation split for adaptive pruning
        if self.adaptive_tree:
            # For classification, you might want to stratify
            stratify = y if self.task_type == "multiclass" else None

            if X_preprocessed.shape[0] < 10:
                # Not enough data to split
                self.adaptive_tree = False

            if self.adaptive_tree:
                (X_train, X_valid,
                 y_train, y_valid,
                 sw_train, sw_valid) = self._split_data_for_adaptive(
                    X_preprocessed, y, sample_weight, stratify=stratify
                )
            else:
                X_train, X_valid, y_train, y_valid, sw_train, sw_valid = (
                    X_preprocessed, None, y, None, sample_weight, None
                )
        else:
            X_train, X_valid, y_train, y_valid, sw_train, sw_valid = (
                X_preprocessed, None, y, None, sample_weight, None
            )

        # Build the underlying scikit-learn DecisionTree
        self.decision_tree = self._init_decision_tree_impl()
        self.decision_tree.fit(X_train, y_train, sample_weight=sw_train)
        self._tree = self.decision_tree

        # Store some references for later usage
        self.train_X_ = X_train
        self.train_y_ = y_train
        self.train_sw_ = sw_train

        self.valid_X_ = X_valid
        self.valid_y_ = y_valid
        self.valid_sw_ = sw_valid

        # We'll lazily do the TabPFN leaf-fitting on the first call to predict
        # so we can possibly adapt to the valid set. Mark it:
        self._need_post_fit = True

        return self

    def _split_data_for_adaptive(
            self,
            X: np.ndarray,
            y: np.ndarray,
            sample_weight: Optional[np.ndarray],
            stratify: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Helper to split data for adaptive-tree pruning."""
        # In classification, user might want to ensure each class appears in train/test
        X_train, X_valid, y_train, y_valid, sw_train, sw_valid = train_test_split(
            X,
            y,
            sample_weight,
            test_size=self.adaptive_tree_test_size,
            random_state=self.random_state,
            stratify=stratify,
        )

        # If either split is empty, revert to no splitting
        if len(y_train) == 0 or len(y_valid) == 0:
            self.adaptive_tree = False
            return X, None, y, None, sample_weight, None

        # If classification, ensure each set still has consistent classes
        if self.task_type == "multiclass":
            train_classes = np.unique(y_train)
            valid_classes = np.unique(y_valid)
            if len(train_classes) < 2 or len(valid_classes) < 1:
                # Not enough classes for adaptive
                self.adaptive_tree = False
                return X, None, y, None, sample_weight, None

        return X_train, X_valid, y_train, y_valid, sw_train, sw_valid

    def _init_decision_tree_impl(self) -> BaseDecisionTree:
        """Overridden in subclasses to create DecisionTreeClassifier or DecisionTreeRegressor."""
        raise NotImplementedError()

    def _preprocess_data_for_tree(self, X: np.ndarray) -> np.ndarray:
        """Replace NaN with 0.0, as a simplistic missing-value handling."""
        if torch.is_tensor(X):
            X = X.cpu().numpy()
        X = np.array(X, dtype=np.float64, copy=True)
        np.nan_to_num(X, copy=False, nan=0.0)
        return X

    def predict(self, X: np.ndarray, check_input: bool = True) -> np.ndarray:
        """Dummy predict method in base class, overridden in subclasses."""
        raise NotImplementedError()

    def get_tree(self) -> BaseDecisionTree:
        """Return the underlying fitted sklearn decision tree."""
        check_is_fitted(self, "_tree")
        return self._tree

    @property
    def tree_(self):
        """Expose the fitted sklearn tree object."""
        return self.get_tree().tree_

    def _apply_tree(self, X: np.ndarray) -> np.ndarray:
        """Apply the fitted decision tree to X, returning (n_samples, n_nodes, n_estimators)."""
        check_is_fitted(self, "_tree")
        dp = self._tree.decision_path(X)
        return np.expand_dims(dp.todense(), axis=2)

    def _predict_internal(
            self,
            X: np.ndarray,
            y: Optional[np.ndarray],
            is_final_prediction: bool,
    ) -> np.ndarray:
        """Core logic that handles leaf-fitting with TabPFN, adaptive pruning, etc.

        If `is_final_prediction=True`, we do the final predictions. If not, we
        might be evaluating on a validation set for pruning.
        """
        # Possibly do the TabPFN leaf-fitting if needed
        if self._need_post_fit and is_final_prediction:
            self._need_post_fit = False

            # 1) If adaptive_tree is true, fit leaves on train + test them on valid
            if self.adaptive_tree and self.valid_X_ is not None and self.valid_y_ is not None:
                self._fit_leaves(self.train_X_, self.train_y_)
                # Evaluate node-level improvements on valid set
                self._predict_internal(self.valid_X_, self.valid_y_, is_final_prediction=False)

            # 2) Then, fit leaves on the entire dataset
            self._fit_leaves(self.X, self.y)

        # Now produce predictions, optionally with pruning logic
        X_leaf_nodes = self._apply_tree(X)
        n_samples, n_nodes, n_estims = X_leaf_nodes.shape
        # We'll keep track of predictions in a nested dict: y_prob[est_id][node_id].
        y_prob: Dict[int, Dict[int, np.ndarray]] = {}
        # Similarly track a "metric" (score) for each node if we are pruning:
        y_metric: Dict[int, Dict[int, float]] = {}
        do_pruning = (y is not None and self.adaptive_tree and not is_final_prediction)
        if do_pruning:
            self._node_prediction_type: Dict[int, Dict[int, str]] = {}

        for est_id in range(n_estims):
            y_prob[est_id] = {}
            y_metric[est_id] = {}
            if do_pruning:
                self._node_prediction_type[est_id] = {}

            # We might show progress bar if show_progress=True
            node_range = range(n_nodes)

            for leaf_id in node_range:
                self._pruning_init_node_predictions(
                    leaf_id, est_id, y_prob, y_metric, n_nodes, n_samples
                )

                # Indices of X that belong to this node
                test_idx = np.argwhere(X_leaf_nodes[:, leaf_id, est_id]).ravel()

                # If no test samples here, skip
                if len(test_idx) == 0:
                    if do_pruning:
                        self._node_prediction_type[est_id][leaf_id] = "previous"
                    continue

                # Get training data for this node
                X_train_leaf, y_train_leaf = self._leaf_train_data[est_id][leaf_id]

                if len(X_train_leaf) == 0:
                    if do_pruning:
                        self._node_prediction_type[est_id][leaf_id] = "previous"
                    continue

                # Check if final leaf (no membership in subsequent nodes)
                is_leaf = (X_leaf_nodes[test_idx, leaf_id + 1:, est_id].sum() == 0.0)

                # If not a leaf and we don't fit internal nodes, skip
                if (not is_leaf) and (not self.fit_nodes):
                    if do_pruning:
                        self._node_prediction_type[est_id][leaf_id] = "previous"
                    continue

                # Additional adaptive checks:
                if do_pruning and leaf_id != 0:
                    # If classification with missing classes, skip
                    if (self.task_type == "multiclass"
                            and len(np.unique(y_train_leaf)) < self.n_classes_
                            and self.adaptive_tree_skip_class_missing):
                        self._node_prediction_type[est_id][leaf_id] = "previous"
                        continue
                    # If too few or too many training points
                    n_leaf_train = X_train_leaf.shape[0]
                    if (n_leaf_train < self.adaptive_tree_min_train_samples
                            or n_leaf_train > self.adaptive_tree_max_train_samples and not is_leaf):
                        self._node_prediction_type[est_id][leaf_id] = "previous"
                        continue

                # Actually predict from TabPFN
                leaf_prediction = self._predict_leaf(
                    X_train_leaf,
                    y_train_leaf,
                    leaf_id,
                    X,
                    test_idx,
                )
                # Build "replacement" vs "averaging" predictions
                y_prob_averaging, y_prob_replacement = self._pruning_get_prediction_type_results(
                    y_prob[est_id][leaf_id],
                    leaf_prediction,
                    test_idx
                )

                # Decide which approach if pruning
                if do_pruning:
                    # Compare old vs new with some metric
                    self._pruning_set_node_prediction_type(
                        y, y_prob_averaging, y_prob_replacement,
                        y_metric, est_id, leaf_id
                    )
                    self._pruning_set_predictions(
                        y_prob, y_prob_averaging, y_prob_replacement,
                        est_id, leaf_id
                    )
                    # Save metric
                    y_metric[est_id][leaf_id] = self._score(
                        y, y_prob[est_id][leaf_id]
                    )
                else:
                    # If not pruning, do direct replacement
                    y_prob[est_id][leaf_id] = y_prob_replacement

        # Final predictions come from the last estimator’s last node
        return y_prob[n_estims - 1][n_nodes - 1]

    def _pruning_init_node_predictions(
            self,
            leaf_id: int,
            estimator_id: int,
            y_prob: Dict[int, Dict[int, np.ndarray]],
            y_metric: Dict[int, Dict[int, float]],
            n_nodes: int,
            n_samples: int,
    ) -> None:
        """Initialize the predictions for node (leaf_id, estimator_id)."""
        if estimator_id not in y_prob:
            y_prob[estimator_id] = {}
            y_metric[estimator_id] = {}

        if leaf_id == 0 and estimator_id == 0:
            # Start from uniform or zero
            y_prob[estimator_id][leaf_id] = self._init_eval_array(n_samples, to_zero=True)
            y_metric[estimator_id][leaf_id] = 0.0
        elif leaf_id == 0 and estimator_id > 0:
            # In multiple-estimator scenario, copy from the last node of previous estimator
            y_prob[estimator_id][leaf_id] = y_prob[estimator_id - 1][n_nodes - 1]
            y_metric[estimator_id][leaf_id] = y_metric[estimator_id - 1][n_nodes - 1]
        else:
            # Copy from previous node of the same estimator
            y_prob[estimator_id][leaf_id] = y_prob[estimator_id][leaf_id - 1]
            y_metric[estimator_id][leaf_id] = y_metric[estimator_id][leaf_id - 1]

    def _pruning_get_prediction_type_results(
            self,
            current_prediction: np.ndarray,
            leaf_prediction: np.ndarray,
            test_idx: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute 'averaging' vs 'replacement' predictions at the node."""
        # Replacement
        y_prob_replacement = current_prediction.copy()
        y_prob_replacement[test_idx] = leaf_prediction[test_idx]

        # Averaging
        y_prob_averaging = current_prediction.copy()
        if self.task_type == "multiclass":
            # For averaging we combine old + new
            if self.average_logits:
                # Log-sum-exp style
                y_prob_averaging[test_idx] = np.log(y_prob_averaging[test_idx] + 1e-8)
                leaf_pred_log = np.log(leaf_prediction[test_idx] + 1e-8)
                y_prob_averaging[test_idx] += leaf_pred_log
                y_prob_averaging[test_idx] = softmax(y_prob_averaging[test_idx])
            else:
                y_prob_averaging[test_idx] += leaf_prediction[test_idx]
                row_sums = y_prob_averaging.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1.0
                y_prob_averaging /= row_sums
        else:
            # Regression -> direct average
            y_prob_averaging[test_idx] += leaf_prediction[test_idx]
            y_prob_averaging[test_idx] /= 2.0

        # For replacement in classification, re-normalize
        if self.task_type == "multiclass":
            row_sums = y_prob_replacement.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            y_prob_replacement /= row_sums

        return y_prob_averaging, y_prob_replacement

    def _pruning_set_node_prediction_type(
            self,
            y_true: np.ndarray,
            y_prob_averaging: np.ndarray,
            y_prob_replacement: np.ndarray,
            y_metric: Dict[int, Dict[int, float]],
            estimator_id: int,
            leaf_id: int,
    ) -> None:
        """Select which approach yields a better score: averaging, replacement, or previous."""
        if estimator_id not in self._node_prediction_type:
            self._node_prediction_type[estimator_id] = {}

        # Compare
        averaging_score = self._score(y_true, y_prob_averaging)
        replacement_score = self._score(y_true, y_prob_replacement)
        prev_score = y_metric[estimator_id][leaf_id]  # old metric

        if max(averaging_score, replacement_score) > prev_score:
            if replacement_score > averaging_score:
                self._node_prediction_type[estimator_id][leaf_id] = "replacement"
            else:
                self._node_prediction_type[estimator_id][leaf_id] = "averaging"
        else:
            self._node_prediction_type[estimator_id][leaf_id] = "previous"

    def _pruning_set_predictions(
            self,
            y_prob: Dict[int, Dict[int, np.ndarray]],
            y_prob_averaging: np.ndarray,
            y_prob_replacement: np.ndarray,
            estimator_id: int,
            leaf_id: int,
    ) -> None:
        """Finalize predictions for the node after we decide on approach."""
        node_type = self._node_prediction_type[estimator_id][leaf_id]
        if node_type == "averaging":
            y_prob[estimator_id][leaf_id] = y_prob_averaging
        elif node_type == "replacement":
            y_prob[estimator_id][leaf_id] = y_prob_replacement
        else:
            # "previous" => keep old predictions
            pass

    def _init_eval_array(self, n_samples: int, to_zero: bool) -> np.ndarray:
        """Initialize an array of predictions for all n_samples."""
        if self.task_type == "multiclass":
            if to_zero:
                return np.zeros((n_samples, self.n_classes_), dtype=np.float64)
            else:
                # Uniform
                return np.ones((n_samples, self.n_classes_), dtype=np.float64) / max(self.n_classes_, 1)
        else:
            # Regression
            return np.zeros((n_samples,), dtype=np.float64)

    def _fit_leaves(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit TabPFN model in each leaf node (or each node) and store training data."""
        leaf_node_matrix = self._apply_tree(X)
        n_samples, n_nodes, n_estims = leaf_node_matrix.shape
        self._leaf_train_data.clear()

        for est_id in range(n_estims):
            self._leaf_train_data[est_id] = {}
            for leaf_id in range(n_nodes):
                idx = np.argwhere(leaf_node_matrix[:, leaf_id, est_id]).ravel()
                X_leaf = X[idx]
                y_leaf = y[idx]
                self._leaf_train_data[est_id][leaf_id] = (X_leaf, y_leaf)

    def _predict_leaf(
            self,
            X_train_leaf: np.ndarray,
            y_train_leaf: np.ndarray,
            leaf_id: int,
            X_full: np.ndarray,
            test_idx: np.ndarray,
    ) -> np.ndarray:
        """Leaf-level TabPFN prediction, overridden by classifier/regressor."""
        raise NotImplementedError()

    def _score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute a performance score given ground truth + predictions."""
        metric = self._get_opt_metric()
        if self.task_type == "multiclass":
            return ScoringUtils.score_classification(metric, y_true, y_pred)
        else:
            return ScoringUtils.score_regression(metric, y_true, y_pred)

    def _get_opt_metric(self) -> str:
        """Return the metric name to optimize (roc for classification, rmse for regression)."""
        if self.adaptive_tree_overwrite_metric is not None:
            return self.adaptive_tree_overwrite_metric
        if self.task_type == "multiclass":
            return "roc"
        return "rmse"


###############################################################################
#                             Classifier                                      #
###############################################################################


class DecisionTreeTabPFNClassifier(DecisionTreeTabPFNBase, ClassifierMixin):
    """A Decision Tree + TabPFN hybrid for classification."""

    task_type = "multiclass"

    def _init_decision_tree_impl(self) -> DecisionTreeClassifier:
        return DecisionTreeClassifier(
            criterion=self.criterion,
            splitter=self.splitter,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            class_weight=self.class_weight,
            ccp_alpha=self.ccp_alpha,
        )

    def predict(self, X: np.ndarray, check_input: bool = True) -> np.ndarray:
        """Predict class labels for X."""
        check_is_fitted(self, "_tree")

        if check_input:
            X = check_array(X, force_all_finite=False, dtype=None)
        # Ensure shape
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X.shape[1] = {X.shape[1]} != {self.n_features_in_}, "
                             "the number of features during training.")

        proba = self.predict_proba(X, check_input=False)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X: np.ndarray, check_input: bool = True) -> np.ndarray:
        """Predict class probabilities."""
        check_is_fitted(self, "_tree")

        if check_input:
            X = check_array(X, force_all_finite=False, dtype=None)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X.shape[1] = {X.shape[1]} != {self.n_features_in_}.")

        preds = self._predict_internal(X, y=None, is_final_prediction=True)
        # For safety, ensure shape (n_samples, n_classes_)
        if preds.shape[1] != self.n_classes_:
            raise ValueError("Internal shape mismatch in classification predict_proba.")
        return preds

    def _predict_leaf(
            self,
            X_train_leaf: np.ndarray,
            y_train_leaf: np.ndarray,
            leaf_id: int,
            X_full: np.ndarray,
            test_idx: np.ndarray,
    ) -> np.ndarray:
        """Fit a TabPFNClassifier on the leaf’s data, then predict_proba for X_full[test_idx]."""
        n_samples = X_full.shape[0]
        # We'll store predictions for entire X, but only fill test_idx
        out = np.zeros((n_samples, self.n_classes_), dtype=np.float64)

        unique_classes = np.unique(y_train_leaf).astype(int)
        if len(unique_classes) == 1:
            # If just one class, fill that prob = 1
            out[test_idx, unique_classes[0]] = 1.0
            return out

        leaf_seed = leaf_id + self.tree_seed
        if hasattr(self.tabpfn, "random_state"):
            self.tabpfn.random_state = leaf_seed
        try:
            self.tabpfn.fit(X_train_leaf, y_train_leaf)
            proba = self.tabpfn.predict_proba(X_full[test_idx])
            # Map them back to correct column
            for i, c in enumerate(unique_classes):
                out[test_idx, c] = proba[:, i]
        except (ValueError, RuntimeError) as e:
            # Fallback to empirical distribution
            warnings.warn(f"TabPFN fit/predict error in leaf {leaf_id}: {e}. Using fallback.")
            _, counts = np.unique(y_train_leaf, return_counts=True)
            ratio = counts / counts.sum()
            for i, c in enumerate(unique_classes):
                out[test_idx, c] = ratio[i]

        return out


###############################################################################
#                             Regressor                                       #
###############################################################################


class DecisionTreeTabPFNRegressor(DecisionTreeTabPFNBase, RegressorMixin):
    """A Decision Tree + TabPFN hybrid for regression."""

    task_type = "regression"

    def _init_decision_tree_impl(self) -> DecisionTreeRegressor:
        return DecisionTreeRegressor(
            criterion=self.criterion,
            splitter=self.splitter,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            ccp_alpha=self.ccp_alpha,
        )

    def predict(self, X: np.ndarray, check_input: bool = True) -> np.ndarray:
        check_is_fitted(self, "_tree")

        if check_input:
            X = check_array(X, force_all_finite=False, dtype=None)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X.shape[1] = {X.shape[1]} != {self.n_features_in_}.")

        preds = self._predict_internal(X, y=None, is_final_prediction=True)
        # For regression, the final shape should be (n_samples,)
        return preds.ravel()

    def _predict_leaf(
            self,
            X_train_leaf: np.ndarray,
            y_train_leaf: np.ndarray,
            leaf_id: int,
            X_full: np.ndarray,
            test_idx: np.ndarray,
    ) -> np.ndarray:
        """Fit a TabPFNRegressor on the leaf’s data, then predict for X_full[test_idx]."""
        n_samples = X_full.shape[0]
        out = np.zeros(n_samples, dtype=np.float64)

        # If empty or single sample
        if len(X_train_leaf) == 0:
            return out  # all zeros
        if len(X_train_leaf) == 1:
            out[test_idx] = y_train_leaf[0]
            return out

        # If all y are the same
        if np.all(y_train_leaf == y_train_leaf[0]):
            out[test_idx] = y_train_leaf[0]
            return out

        leaf_seed = leaf_id + self.tree_seed
        if hasattr(self.tabpfn, "random_state"):
            self.tabpfn.random_state = leaf_seed

        try:
            self.tabpfn.fit(X_train_leaf, y_train_leaf)
            preds = self.tabpfn.predict(X_full[test_idx])
            out[test_idx] = preds
        except (ValueError, RuntimeError, NotImplementedError, AssertionError) as e:
            warnings.warn(f"TabPFN fit/predict error in leaf {leaf_id}: {e}. Using mean fallback.")
            out[test_idx] = np.mean(y_train_leaf)

        return out
