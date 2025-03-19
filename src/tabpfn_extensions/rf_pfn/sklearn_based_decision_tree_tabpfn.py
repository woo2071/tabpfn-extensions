# Copyright (c) Prior Labs GmbH 2025.
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import random
import warnings
from typing import Any

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import (
    BaseDecisionTree,
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)

from tabpfn_extensions.utils import softmax


class ScoringUtils:
    """Utility class for scoring classification and regression models."""

    @staticmethod
    def score_classification(metric_name, y_true=None, y_proba=None):
        """Score classification results with the given metric.

        Parameters
        ----------
        metric_name : str
            Name of the metric to use, e.g. "roc"
        y_true : array-like, optional
            True labels
        y_proba : array-like, optional
            Predicted probabilities

        Returns:
        -------
        float
            Score value
        """
        if metric_name == "roc":
            # Dummy ROC-like measure
            return 0.5
        return 0.0

    @staticmethod
    def score_regression(metric_name, y_true, y_pred):
        """Score regression results with the given metric.

        Parameters
        ----------
        metric_name : str
            Name of the metric to use, e.g. "rmse"
        y_true : array-like
            True values
        y_pred : array-like
            Predicted values

        Returns:
        -------
        float
            Score value
        """
        if metric_name == "rmse":
            return np.sqrt(np.mean((y_true - y_pred) ** 2))
        return 9999.0


###############################################################################
#                             BASE DECISION TREE                              #
###############################################################################


class DecisionTreeTabPFNBase(BaseDecisionTree, BaseEstimator):
    """Abstract base class combining a scikit-learn Decision Tree with TabPFN at the leaves.

    This class:
      • Inherits from sklearns BaseDecisionTree to leverage the standard decision-tree splitting.
      • Holds references to a TabPFN (Classifier or Regressor) to fit leaf nodes (or all nodes).
      • Provides adaptive pruning logic (optional) via `adaptive_tree`.

    Subclasses:
      • DecisionTreeTabPFNClassifier
      • DecisionTreeTabPFNRegressor

    Parameters
    ----------
    tabpfn : object
        A TabPFNClassifier or TabPFNRegressor instance.
    criterion : str
        The function to measure the quality of a split. (See sklearn docs).
    splitter : str
        The strategy used to choose the split at each node. (e.g. "best" or "random")
    max_depth : int, optional
        The maximum depth of the tree. (See sklearn docs).
    min_samples_split : int
        The minimum number of samples required to split an internal node.
    min_samples_leaf : int
        The minimum number of samples required to be at a leaf node.
    min_weight_fraction_leaf : float
        The minimum weighted fraction of the sum total of weights required to be at a leaf node.
    max_features : int, float, str or None
        The number of features to consider when looking for the best split.
    random_state : int, RandomState instance or None
        Controls the randomness of the estimator.
    max_leaf_nodes : int or None
        Grow a tree with max_leaf_nodes in best-first fashion.
    min_impurity_decrease : float
        A node will be split if this split induces a decrease of the impurity greater or equal to this value.
    ccp_alpha : non-negative float
        Complexity parameter used for Minimal Cost-Complexity Pruning.
    monotonic_cst : Any
        Optional monotonicity constraints (depending on sklearn version).
    categorical_features : list of int, optional
        Indices of categorical features for TabPFN usage (if any).
    verbose : bool or int
        Verbosity level.
    show_progress : bool
        Whether to show progress bars for leaf/node fitting using TabPFN.
    fit_nodes : bool
        Whether to fit TabPFN at internal nodes (True) or only final leaves (False).
    tree_seed : int
        Used to set seeds for TabPFN fitting in each node.
    adaptive_tree : bool
        Whether to do adaptive node-by-node pruning using a hold-out strategy.
    adaptive_tree_min_train_samples : int
        Minimum number of training samples required to fit a TabPFN in a node.
    adaptive_tree_max_train_samples : int
        Maximum number of training samples above which a node might be pruned if not a final leaf.
    adaptive_tree_min_valid_samples_fraction_of_train : float
        Fraction controlling the minimum valid/test points to consider a node for re-fitting.
    adaptive_tree_overwrite_metric : str or None
        If set, overrides the default metric for pruning. E.g., "roc" or "rmse".
    adaptive_tree_test_size : float
        Fraction of data to hold out for adaptive pruning if no separate valid set is provided.
    average_logits : bool
        Whether to average logits (True) or probabilities (False) for the "averaging" method in adaptive_tree.
    adaptive_tree_skip_class_missing : bool
        If True, skip re-fitting if the nodes train set does not contain all classes (for classification).
    """

    task_type: str = None

    def __init__(
        self,
        *,
        # Decision Tree arguments
        criterion: str = "gini",
        splitter: str = "best",
        max_depth: int | None = None,
        min_samples_split: int = 1000,
        min_samples_leaf: int = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_features=None,
        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease: float = 0.0,
        class_weight=None,  # only used in classification
        ccp_alpha: float = 0.0,
        monotonic_cst=None,
        # TabPFN argument
        tabpfn=None,
        categorical_features=None,
        verbose=False,
        show_progress=False,
        fit_nodes=True,
        tree_seed=0,
        adaptive_tree=True,
        adaptive_tree_min_train_samples=50,
        adaptive_tree_max_train_samples=2000,
        adaptive_tree_min_valid_samples_fraction_of_train=0.2,
        adaptive_tree_overwrite_metric=None,
        adaptive_tree_test_size: float = 0.2,
        average_logits=True,
        adaptive_tree_skip_class_missing=True,
    ):
        # Validate that tabpfn is not None and appropriate
        self._validate_tabpfn_init(tabpfn)

        # Collect recognized arguments
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

        self.n_classes_ = 0
        self.classes_ = []

        # We store leaf nodes and training data in them for TabPFN
        self.leaf_nodes = []
        self.leaf_train_data = {}

        # The actual sklearn decision tree
        self.decision_tree = None

        # Keep label encoder for classification
        self.label_encoder_ = LabelEncoder()

        # Next lines handle possible differences in sklearn versions, specifically monotonic_cst
        optional_args_filtered = {}
        # Checking if monotonic_cst is recognized by current sklearn base tree
        if BaseDecisionTree.__init__.__code__.co_varnames.__contains__("monotonic_cst"):
            optional_args_filtered["monotonic_cst"] = monotonic_cst

        # Initialize the underlying DecisionTree with super() call
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
            **optional_args_filtered,
        )

        # If the user gave a TabPFN, we do not want it to have a random_state set forcibly
        # because we handle seeds ourselves at each node
        if self.tabpfn:
            self.tabpfn.random_state = None

        # Internal flags/structures
        self.todo_post_fit = False

    def _validate_tabpfn_init(self, tabpfn):
        """Ensure the `tabpfn` argument is not None during __init__."""
        if tabpfn is None:
            raise ValueError(
                "tabpfn parameter cannot be None. Provide a TabPFNClassifier or TabPFNRegressor instance.",
            )

    def _validate_tabpfn_runtime(self):
        """Optionally refine checks at runtime, if needed."""
        if self.tabpfn is None:
            raise ValueError("TabPFN was None at runtime - cannot proceed.")

    def _more_tags(self) -> dict[str, Any]:
        """Additional sklearn tags.
        We keep this for older sklearn versions. It can be replaced or combined
        with _get_tags() in newer versions, but it’s recognized by older versions.
        """
        return {"multilabel": True, "allow_nan": True}

    def __sklearn_tags__(self):
        """Official sklearn method for new versions.
        Return tags dictionary. We adapt the old code logic
        plus new codes docstring approach.
        """
        tags = super().__sklearn_tags__()
        tags["allow_nan"] = True
        # Usually "estimator_type" gets set automatically, but we can ensure it:
        if self.task_type == "multiclass":
            tags["estimator_type"] = "classifier"
        else:
            tags["estimator_type"] = "regressor"
        return tags

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
        check_input: bool = True,
    ):
        """Fit the DecisionTree + TabPFN model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
        check_input : bool, default=True
            Whether to check the input array(s).

        Returns:
        -------
        self : object
            Fitted estimator.
        """
        return self._fit(X, y, sample_weight=sample_weight, check_input=check_input)

    def _fit(
        self,
        X,
        y,
        sample_weight=None,
        check_input=True,
        missing_values_in_feature_mask=None,
    ):
        """Method that fits the DecisionTree-TabPFN model on X, y.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features
        y : array-like of shape (n_samples,)
            Target labels/values
        sample_weight : array-like of shape (n_samples,)
            If provided, sample weights for each sample
        check_input : bool
            Whether to check inputs
        missing_values_in_feature_mask : unused in old code, optional

        Returns:
        -------
        self
        """
        # Make sure tabpfn is valid
        self._validate_tabpfn_runtime()

        # Possibly randomize tree_seed if not set
        if self.tree_seed == 0:
            self.tree_seed = random.randint(1, 10000)

        # Old code used a direct approach to handle input
        if check_input:
            # We can do minimal checks or rely on sklearns own checks
            # Typically, sklearn handles it in super().fit
            pass

        # Convert torch tensor -> numpy if needed
        X_preprocessed = self.preprocess_data_for_tree(X)

        if sample_weight is None:
            sample_weight = np.ones((X_preprocessed.shape[0],))

        # Setup classes_ or n_classes_ if needed
        if self.task_type == "multiclass":
            # Classification
            self.classes_ = np.unique(y)
            self.n_classes_ = len(self.classes_)
        else:
            # Regression
            self.n_classes_ = 1  # Not used for numeric tasks, but we keep a variable

        # Possibly label-encode y in classification scenario
        # The old code just took y = y.copy() then counted unique
        # Well keep it consistent with old approach
        y_ = y.copy()

        if self.task_type == "multiclass":
            # At old code, we had: self.n_classes_ = int(y.max().iloc[0] + 1)
            # but that was for a very specific situation. Well just rely on classes_ array.
            # (the old code used: y = pd.DataFrame(y).reset_index().drop("index", axis=1) if y was a Series.)
            pass

        # If adaptive_tree is on, we do a train/valid split
        if self.adaptive_tree:
            # Only stratify if classification
            stratify = y_ if (self.task_type == "multiclass") else None

            # Check if we can even do a split (like old code did)
            # The old code turned off adaptive_tree if a class had only 1 sample
            if self.task_type == "multiclass":
                _, counts = np.unique(y_, return_counts=True)
                if counts.min() == 1:
                    self.adaptive_tree = False

            if self.adaptive_tree:
                (
                    X_train,
                    X_valid,
                    X_preproc_train,
                    X_preproc_valid,
                    y_train,
                    y_valid,
                    sw_train,
                    sw_valid,
                ) = train_test_split(
                    X,
                    X_preprocessed,
                    y_,
                    sample_weight,
                    test_size=self.adaptive_tree_test_size,
                    random_state=self.random_state,
                    stratify=stratify,
                )

                # Another check from old code: if classification and train/valid lost classes -> disable
                if self.task_type == "multiclass" and len(np.unique(y_train)) != len(
                    np.unique(y_valid),
                ):
                    self.adaptive_tree = False
            else:
                # If we got disabled, keep everything as training
                X_train, X_preproc_train, y_train, sw_train = (
                    X,
                    X_preprocessed,
                    y_,
                    sample_weight,
                )
                X_valid = X_preproc_valid = y_valid = sw_valid = None
        else:
            # Not adaptive, everything is train
            X_train, X_preproc_train, y_train, sw_train = (
                X,
                X_preprocessed,
                y_,
                sample_weight,
            )
            X_valid = X_preproc_valid = y_valid = sw_valid = None

        # Initialize actual sklearn decision tree
        self._tree = self.init_decision_tree()
        self._tree.fit(X_preproc_train, y_train, sample_weight=sw_train)

        # Store some references for later usage
        self.X = X
        self.y = y_
        self.train_X = X_train
        self.train_X_preprocessed = X_preproc_train
        self.train_y = y_train
        self.train_sample_weight = sw_train

        if self.adaptive_tree:
            self.valid_X = X_valid
            self.valid_X_preprocessed = X_preproc_valid
            self.valid_y = y_valid
            self.valid_sample_weight = sw_valid

        # Mark that we have to run some leaf-fitting logic if needed
        self.todo_post_fit = True
        if self.verbose:
            self.post_fit()

        return self

    def fit_leafs(self, train_X, train_y):
        """Fit a TabPFN model in each leaf node (or each node, if self.fit_nodes=True).

        Old code logic: we get `leaf_nodes` for each sample, storing them in self.leaf_train_data.
        """
        self.leaf_train_data = {}
        # Get leaf membership for each sample
        leaf_node_matrix, bootstrap_X, bootstrap_y = self.apply_tree_train(
            train_X,
            train_y,
        )
        self.leaf_nodes = leaf_node_matrix

        # For each estimator_id (like old code, though we usually only have 1 estimator in a single decision tree),
        # store the samples in each leaf
        n_samples, n_nodes, n_estims = leaf_node_matrix.shape

        for estimator_id in range(n_estims):
            self.leaf_train_data[estimator_id] = {}
            for leaf_id in range(n_nodes):
                indices = np.argwhere(
                    leaf_node_matrix[:, leaf_id, estimator_id],
                ).ravel()
                X_train_samples = np.take(train_X, indices, axis=0)
                y_train_samples = np.array(np.take(train_y, indices, axis=0)).ravel()
                self.leaf_train_data[estimator_id][leaf_id] = (
                    X_train_samples,
                    y_train_samples,
                )

    def get_tree(self):
        """Return the underlying fitted sklearn decision tree."""
        return self._tree

    def apply_tree(self, X) -> np.ndarray:
        """Apply the tree to X to get a matrix of shape (N_samples, N_nodes, N_estimators).
        In the old code, we had only 1 tree, but the shape was expanded.
        """
        decision_path = self.get_tree().decision_path(X)
        return np.expand_dims(decision_path.todense(), 2)

    def apply_tree_train(self, X, y):
        """Apply the tree for training data, returning leaf membership plus X, y unchanged."""
        return self.apply_tree(X), X, y

    def init_decision_tree(self):
        """This is overridden in the child classes for either classifier or regressor."""
        raise NotImplementedError("init_decision_tree must be implemented in subclass.")

    def post_fit(self):
        """Hook after fit. Old code just printed something if verbose."""
        pass

    def preprocess_data_for_tree(self, X: np.ndarray) -> np.ndarray:
        """Handle missing data prior to feeding into the decision tree.
        In old code, we just replaced NaNs with 0.0, which is simplistic but consistent with the old approach.
        """
        if torch.is_tensor(X):
            X = X.cpu().numpy()
        X = np.array(X, dtype=np.float64)
        np.nan_to_num(X, copy=False, nan=0.0)
        return X

    def predict_(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        check_input: bool = True,
    ):
        """Internal method used to produce probability/regression predictions with adaptive tree pruning.

        If y is given and we are adaptive_tree, we do node-level pruning based on performance on (X,y).
        Otherwise, we just produce final predictions from the best-known node approach.
        """
        # This logic triggers if we havent done the final leaf fit
        if self.todo_post_fit:
            self.todo_post_fit = False
            if self.adaptive_tree:
                # Old code: do a partial leaf fit on train, check performance on valid
                if (
                    self.verbose
                    and hasattr(self, "valid_X")
                    and self.valid_X is not None
                ):
                    pass
                self.fit_leafs(self.train_X, self.train_y)
                if (
                    hasattr(self, "valid_X")
                    and self.valid_X is not None
                    and self.valid_y is not None
                ):
                    # Force a pass to evaluate node performance
                    self.predict_(self.valid_X, self.valid_y, check_input=False)
            # Finally, fit all leaves on the entire dataset
            self.fit_leafs(self.X, self.y)

        # Set categorical features if needed
        if self.tabpfn is not None:
            self.tabpfn.categorical_features_indices = self.categorical_features

        # Get membership of X in each node
        X_preprocessed = self.preprocess_data_for_tree(X)
        X_leaf_nodes = self.apply_tree(X_preprocessed)
        N_samples, N_nodes, N_estimators = X_leaf_nodes.shape

        # Well store intermediate predictions in y_prob, and possibly metrics in y_metric
        y_prob, y_metric = {}, {}
        for est_id in range(N_estimators):
            y_prob[est_id] = {}
            y_metric[est_id] = {}

        # If we’re pruning, we track how each node is updated
        prune_leafs_and_nodes = (y is not None) and self.adaptive_tree
        if prune_leafs_and_nodes:
            self.node_prediction_type = {}
            for est_id in range(N_estimators):
                self.node_prediction_type[est_id] = {}

        for est_id in range(N_estimators):
            if self.show_progress:
                import tqdm.auto

                node_iter = tqdm.auto.tqdm(range(N_nodes))
            else:
                node_iter = range(N_nodes)

            for leaf_id in node_iter:
                # Initialize predictions at this node
                self._pruning_init_node_predictions(
                    leaf_id,
                    est_id,
                    y_prob,
                    y_metric,
                    N_nodes,
                    N_samples,
                )
                # If first leaf in a subsequent estimator, skip
                if est_id > 0 and leaf_id == 0:
                    continue

                # Gather test-sample indices that belong to this leaf
                test_sample_indices = np.argwhere(
                    X_leaf_nodes[:, leaf_id, est_id],
                ).ravel()

                # Gather training samples that belong to this leaf
                X_train_leaf, y_train_leaf = self.leaf_train_data[est_id][leaf_id]

                # If no training or no test samples in node, skip
                if (X_train_leaf.shape[0] == 0) or (len(test_sample_indices) == 0):
                    if prune_leafs_and_nodes:
                        self.node_prediction_type[est_id][leaf_id] = "previous"
                    continue

                # Check if it is a final leaf or not
                # The old code sets is_leaf = sum of subsequent node memberships
                is_leaf = (
                    X_leaf_nodes[test_sample_indices, leaf_id + 1 :, est_id].sum()
                    == 0.0
                )

                # If its not a leaf and we are not fitting nodes, skip
                if (
                    (not is_leaf)
                    and (not self.fit_nodes)
                    and not (leaf_id == 0 and self.adaptive_tree)
                ):
                    if prune_leafs_and_nodes:
                        self.node_prediction_type[est_id][leaf_id] = "previous"
                    continue

                # Additional adaptive checks: skip if not enough data
                if self.adaptive_tree and leaf_id != 0:
                    # Possibly skip if node was pruned previously
                    if (y is None) and (
                        self.node_prediction_type[est_id][leaf_id] == "previous"
                    ):
                        continue
                    # Skip if classification missing some class
                    if (
                        self.task_type == "multiclass"
                        and len(np.unique(y_train_leaf)) < self.n_classes_
                        and self.adaptive_tree_skip_class_missing
                    ):
                        self.node_prediction_type[est_id][leaf_id] = "previous"
                        continue

                    # Skip if too few or too many training points, etc.
                    if (
                        (X_train_leaf.shape[0] < self.adaptive_tree_min_train_samples)
                        or (
                            len(test_sample_indices)
                            < self.adaptive_tree_min_valid_samples_fraction_of_train
                            * self.adaptive_tree_min_train_samples
                        )
                        or (
                            X_train_leaf.shape[0] > self.adaptive_tree_max_train_samples
                            and not is_leaf
                        )
                    ):
                        if prune_leafs_and_nodes and self.verbose:
                            pass
                        if prune_leafs_and_nodes:
                            self.node_prediction_type[est_id][leaf_id] = "previous"
                        continue

                # If verbose, we show shape info
                if self.verbose:
                    if y is not None:
                        pass
                    else:
                        pass

                # Do the actual leaf-level TabPFN prediction
                leaf_prediction = self.predict_leaf_(
                    X_train_leaf,
                    y_train_leaf,
                    leaf_id,
                    X_preprocessed,
                    test_sample_indices,
                )

                # Evaluate both “averaging” and “replacement” for adaptive pruning
                y_prob_averaging, y_prob_replacement = (
                    self._pruning_get_prediction_type_results(
                        y_prob,
                        leaf_prediction,
                        test_sample_indices,
                        est_id,
                        leaf_id,
                    )
                )

                # If adaptive tree, decide how to pick among “averaging”, “replacement”, or “previous”
                if self.adaptive_tree:
                    if y is not None:
                        self._pruning_set_node_prediction_type(
                            y,
                            y_prob_averaging,
                            y_prob_replacement,
                            y_metric,
                            est_id,
                            leaf_id,
                        )
                    self._pruning_set_predictions(
                        y_prob,
                        y_prob_averaging,
                        y_prob_replacement,
                        est_id,
                        leaf_id,
                    )
                    if y is not None:
                        y_metric[est_id][leaf_id] = self._score(
                            y,
                            y_prob[est_id][leaf_id],
                        )
                        if self.verbose:
                            pass
                else:
                    # If not adaptive, we just do “replacement”
                    y_prob[est_id][leaf_id] = y_prob_replacement

        # Return final predictions from the last estimators last node
        return y_prob[N_estimators - 1][N_nodes - 1]

    def _pruning_init_node_predictions(
        self,
        leaf_id: int,
        estimator_id: int,
        y_prob: dict,
        y_metric: dict,
        N_nodes: int,
        N_samples: int,
    ):
        """Initialize y_prob / y_metric for the current node,
        based on the old approach where we copy forward from the previous node or estimator.
        """
        if estimator_id == 0 and leaf_id == 0:
            y_prob[0][0] = self.init_eval_prob(N_samples, to_zero=True)
            y_metric[0][0] = 0
        elif leaf_id == 0 and estimator_id > 0:
            # If first leaf of new tree, carry from last node of previous estimator
            y_prob[estimator_id][leaf_id] = y_prob[estimator_id - 1][N_nodes - 1]
            y_metric[estimator_id][leaf_id] = y_metric[estimator_id - 1][N_nodes - 1]
        else:
            # use last leaf of same estimator
            y_prob[estimator_id][leaf_id] = y_prob[estimator_id][leaf_id - 1]
            y_metric[estimator_id][leaf_id] = y_metric[estimator_id][leaf_id - 1]

    def _pruning_get_prediction_type_results(
        self,
        y_eval_prob: dict,
        leaf_prediction: np.ndarray,
        test_sample_indices: np.ndarray,
        estimator_id: int,
        leaf_id: int,
    ):
        """The old approach to produce y_prob_averaging and y_prob_replacement.
        For classification, we do either probability addition or logit addition
        For regression, we do a simple average.
        """
        y_prob_current = y_eval_prob[estimator_id][leaf_id]
        y_prob_replacement = np.copy(y_prob_current)
        # “replacement” -> directly set the new leaf prediction
        y_prob_replacement[test_sample_indices] = leaf_prediction[test_sample_indices]
        if self.task_type == "multiclass":
            # Normalize
            row_sums = y_prob_replacement.sum(axis=1, keepdims=True)
            # Avoid division by zero
            row_sums[row_sums == 0] = 1.0
            y_prob_replacement /= row_sums
        # “averaging” -> combine old predictions with new
        y_prob_averaging = np.copy(y_prob_current)

        if self.task_type == "multiclass":
            if self.average_logits:
                # convert old + new to log, sum them, softmax
                y_prob_averaging[test_sample_indices] = np.log(
                    y_prob_averaging[test_sample_indices] + 1e-6,
                )
                leaf_pred_log = np.log(leaf_prediction[test_sample_indices] + 1e-6)
                y_prob_averaging[test_sample_indices] += leaf_pred_log
                y_prob_averaging[test_sample_indices] = softmax(
                    y_prob_averaging[test_sample_indices],
                )
            else:
                # average the probabilities directly
                y_prob_averaging[test_sample_indices] += leaf_prediction[
                    test_sample_indices
                ]
                # renormalize
                row_sums = y_prob_averaging.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1.0
                y_prob_averaging /= row_sums
        else:
            # regression -> average
            y_prob_averaging[test_sample_indices] += leaf_prediction[
                test_sample_indices
            ]
            y_prob_averaging[test_sample_indices] /= 2.0

        return y_prob_averaging, y_prob_replacement

    def _pruning_set_node_prediction_type(
        self,
        y_true: np.ndarray,
        y_prob_averaging: np.ndarray,
        y_prob_replacement: np.ndarray,
        y_metric: dict,
        estimator_id: int,
        leaf_id: int,
    ):
        """Decide which approach is better: “averaging” vs “replacement” vs “previous”,
        based on the old code’s approach that compares their scores to the last known metric.
        """
        averaging_score = self._score(y_true, y_prob_averaging)
        replacement_score = self._score(y_true, y_prob_replacement)
        prev_score = y_metric[estimator_id][leaf_id - 1] if (leaf_id > 0) else 0

        if (leaf_id == 0) or (max(averaging_score, replacement_score) > prev_score):
            # pick whichever is better
            if replacement_score > averaging_score:
                prediction_type = "replacement"
            else:
                prediction_type = "averaging"
        else:
            prediction_type = "previous"

        self.node_prediction_type[estimator_id][leaf_id] = prediction_type
        if self.verbose:
            pass

    def _pruning_set_predictions(
        self,
        y_prob: dict,
        y_prob_averaging: np.ndarray,
        y_prob_replacement: np.ndarray,
        estimator_id: int,
        leaf_id: int,
    ):
        """Given the chosen node_prediction_type, set the final node predictions."""
        node_type = self.node_prediction_type[estimator_id][leaf_id]
        if node_type == "averaging":
            y_prob[estimator_id][leaf_id] = y_prob_averaging
        elif node_type == "replacement":
            y_prob[estimator_id][leaf_id] = y_prob_replacement
        else:
            # “previous”
            y_prob[estimator_id][leaf_id] = y_prob[estimator_id][leaf_id - 1]

    def _score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Score the predictions. The old code calls out to scoring_utils with "roc"/"rmse" etc."""
        metric = self.get_optimize_metric()
        if self.task_type == "multiclass":
            # classification
            return ScoringUtils.score_classification(metric, y_true, y_pred)
        else:
            # regression
            return ScoringUtils.score_regression(metric, y_true, y_pred)

    def get_optimize_metric(self) -> str:
        """Return which metric name we’re using.
        Old code defaulted to roc for classification, rmse for regression
        unless overwritten.
        """
        if self.adaptive_tree_overwrite_metric is not None:
            return self.adaptive_tree_overwrite_metric
        if self.task_type == "multiclass":
            return "roc"
        else:
            return "rmse"

    def predict_leaf_(
        self,
        X_train_samples: np.ndarray,
        y_train_samples: np.ndarray,
        leaf_id: int,
        X: np.ndarray,
        indices: np.ndarray,
    ) -> np.ndarray:
        """Each subclass must implement how to actually call TabPFN at the leaf."""
        raise NotImplementedError("Subclass must implement predict_leaf_.")

    def init_eval_prob(self, n_samples: int, to_zero: bool = False) -> np.ndarray:
        """Initialize an array of predictions for the entire dataset,
        either for classification or regression.
        """
        if self.task_type == "multiclass":
            if to_zero:
                return np.zeros((n_samples, self.n_classes_))
            else:
                return np.ones((n_samples, self.n_classes_)) / self.n_classes_
        else:
            # regression
            return np.zeros((n_samples,))


###############################################################################
#                          CLASSIFIER SUBCLASS                                #
###############################################################################


class DecisionTreeTabPFNClassifier(DecisionTreeTabPFNBase, ClassifierMixin):
    """Decision tree that uses TabPFNClassifier at the leaves.

    Inherits the old code’s logic from `DecisionTreeTabPFNBase`.
    """

    task_type: str = "multiclass"

    def init_decision_tree(self) -> DecisionTreeClassifier:
        """Create the sklearn DecisionTreeClassifier with parameters we stored."""
        return DecisionTreeClassifier(
            criterion=self.criterion,
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
            splitter=self.splitter,
        )

    def predict_leaf_(
        self,
        X_train_samples: np.ndarray,
        y_train_samples: np.ndarray,
        leaf_id: int,
        X: np.ndarray,
        indices: np.ndarray,
    ) -> np.ndarray:
        """Fit a TabPFNClassifier on the leaf’s train data and predict_proba
        for the given indices in X.
        """
        y_eval_prob = self.init_eval_prob(X.shape[0], to_zero=True)
        classes_in_leaf = np.unique(y_train_samples).astype(int)

        # If only one class is present, we do an immediate fill
        if len(classes_in_leaf) == 1:
            y_eval_prob[indices, classes_in_leaf[0]] = 1.0
            return y_eval_prob

        # Otherwise, we attempt to fit TabPFN
        leaf_seed = leaf_id + self.tree_seed
        try:
            self.tabpfn.random_state = leaf_seed
            self.tabpfn.fit(X_train_samples, y_train_samples)
            # Predict on the relevant samples
            proba = self.tabpfn.predict_proba(X[indices])
            # Place these in the correct columns
            for i, c in enumerate(classes_in_leaf):
                y_eval_prob[indices, c] = proba[:, i]
        except ValueError as e:
            # Handle the old code’s “constant features” scenario, or re-raise
            if (len(e.args) == 0) or (
                e.args[0]
                != "All features are constant and would have been removed! Unable to predict using TabPFN."
            ):
                raise e
            warnings.warn(
                "One leaf node has only constant features for TabPFN. Returning ratio of target classes.",
                stacklevel=2,
            )
            # fallback: use class ratio from training
            _, cnts = np.unique(y_train_samples, return_counts=True)
            ratio = cnts / cnts.sum()
            for i, c in enumerate(classes_in_leaf):
                y_eval_prob[indices, c] = ratio[i]

        # Return the probability matrix for all samples
        return y_eval_prob

    def predict(self, X: np.ndarray, check_input: bool = True) -> np.ndarray:
        """Predict classes (argmax over predict_proba)."""
        # Simply do argmax on predict_proba
        proba = self.predict_proba(X, check_input=check_input)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X: np.ndarray, check_input: bool = True) -> np.ndarray:
        """Predict class probabilities for X using the TabPFN leaves."""
        return self.predict_(X, check_input=check_input)

    def post_fit(self):
        """Hook after the decision tree is fitted."""
        # old code had pass, or an optional debug print.
        if self.verbose:
            pass


###############################################################################
#                           REGRESSOR SUBCLASS                                #
###############################################################################


class DecisionTreeTabPFNRegressor(DecisionTreeTabPFNBase, RegressorMixin):
    """Decision tree that uses TabPFNRegressor at the leaves.

    Inherits old code logic from `DecisionTreeTabPFNBase`.
    """

    task_type: str = "regression"

    def __init__(
        self,
        *,
        criterion="squared_error",
        splitter="best",
        max_depth=None,
        min_samples_split=1000,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        ccp_alpha=0.0,
        monotonic_cst=None,
        tabpfn=None,
        categorical_features=None,
        verbose=False,
        show_progress=False,
        fit_nodes=True,
        tree_seed=0,
        adaptive_tree=True,
        adaptive_tree_min_train_samples=50,
        adaptive_tree_max_train_samples=2000,
        adaptive_tree_min_valid_samples_fraction_of_train=0.2,
        adaptive_tree_overwrite_metric=None,
        adaptive_tree_test_size=0.2,
        average_logits=True,
        adaptive_tree_skip_class_missing=True,
    ):
        # Just call super with these parameters
        super().__init__(
            tabpfn=tabpfn,
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            random_state=random_state,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            ccp_alpha=ccp_alpha,
            monotonic_cst=monotonic_cst,
            categorical_features=categorical_features,
            verbose=verbose,
            show_progress=show_progress,
            fit_nodes=fit_nodes,
            tree_seed=tree_seed,
            adaptive_tree=adaptive_tree,
            adaptive_tree_min_train_samples=adaptive_tree_min_train_samples,
            adaptive_tree_max_train_samples=adaptive_tree_max_train_samples,
            adaptive_tree_min_valid_samples_fraction_of_train=(
                adaptive_tree_min_valid_samples_fraction_of_train
            ),
            adaptive_tree_overwrite_metric=adaptive_tree_overwrite_metric,
            adaptive_tree_test_size=adaptive_tree_test_size,
            average_logits=average_logits,
            adaptive_tree_skip_class_missing=adaptive_tree_skip_class_missing,
        )

    def init_decision_tree(self) -> DecisionTreeRegressor:
        """Create the underlying sklearn DecisionTreeRegressor with stored parameters."""
        return DecisionTreeRegressor(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            ccp_alpha=self.ccp_alpha,
            splitter=self.splitter,
        )

    def predict_leaf_(
        self,
        X_train_samples: np.ndarray,
        y_train_samples: np.ndarray,
        leaf_id: int,
        X: np.ndarray,
        indices: np.ndarray,
    ) -> np.ndarray:
        """Fit TabPFNRegressor on the nodes data, then predict for the relevant test samples.

        Returns an array of shape (N_samples,) with predictions for the entire X,
        but we only fill `indices`.
        """
        # Well store predictions in a 1D array
        y_eval = np.zeros(X.shape[0], dtype=float)

        # If no training data or just 1 sample, default to mean
        if len(X_train_samples) < 1:
            warnings.warn(
                f"Leaf {leaf_id} has zero training samples. Returning 0.0 predictions.",
                stacklevel=2,
            )
            return y_eval
        elif len(X_train_samples) == 1:
            y_eval[indices] = y_train_samples[0]
            return y_eval

        # If all y are identical, just return that constant
        if np.all(y_train_samples == y_train_samples[0]):
            y_eval[indices] = y_train_samples[0]
            return y_eval

        # Fit the TabPFNRegressor
        leaf_seed = leaf_id + self.tree_seed
        try:
            self.tabpfn.random_state = leaf_seed
            self.tabpfn.fit(X_train_samples, y_train_samples)
            # Now predict
            preds = self.tabpfn.predict(X[indices])
            y_eval[indices] = preds
        except (ValueError, RuntimeError, NotImplementedError, AssertionError) as e:
            warnings.warn(
                f"TabPFN fit/predict failed at leaf {leaf_id}: {e}. Using mean fallback.",
                stacklevel=2,
            )
            y_eval[indices] = np.mean(y_train_samples)

        return y_eval

    def predict(self, X: np.ndarray, check_input: bool = True) -> np.ndarray:
        """Predict regression values using the TabPFN leaves."""
        return self.predict_(X, check_input=check_input)

    def predict_full(self, X: np.ndarray) -> np.ndarray:
        """(Old code’s convenience method) Predict with no input checks."""
        return self.predict_(X, check_input=False)

    def post_fit(self):
        """Hook after regressor decision tree is fitted."""
        if self.verbose:
            pass
