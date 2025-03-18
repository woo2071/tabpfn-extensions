#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

"""Decision tree implementation that uses TabPFN at the leaves.

This module provides scikit-learn compatible decision trees that use TabPFN models
at the leaves for improved prediction performance. It defines two main classes:

- DecisionTreeTabPFNBase: Abstract base class with common functionality
- DecisionTreeTabPFNClassifier: Classification tree with TabPFN leaves
- DecisionTreeTabPFNRegressor: Regression tree with TabPFN leaves

These implementations can be used as standalone models or as building blocks
for the random forest implementations in SklearnBasedRandomForestTabPFN.py.
"""

from __future__ import annotations

import random
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import BaseDecisionTree, DecisionTreeClassifier, DecisionTreeRegressor

if TYPE_CHECKING:
    from numpy.typing import NDArray


class DecisionTreeTabPFNBase(BaseEstimator):
    """Base class for TabPFN-enhanced decision trees.

    This class combines decision trees with TabPFN models at the leaves,
    providing a hybrid approach that leverages the interpretability of
    decision trees with the predictive power of TabPFN.

    Note: This is an abstract base class that should not be used directly.
    Use the classifier or regressor implementations instead.
    """

    task_type = None

    def __init__(
        self,
        *,
        tabpfn,
        criterion="gini",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        ccp_alpha=0.0,
        monotonic_cst=None,
        categorical_features=None,
        X_valid=None,
        y_valid=None,
        verbose=0,
        show_progress=True,
        fit_nodes=False,
        tree_seed=None,
        adaptive_tree=False,
        adaptive_tree_min_train_samples=10,
        adaptive_tree_max_train_samples=10000,
        adaptive_tree_min_valid_samples_fraction_of_train=0.5,
        adaptive_tree_patience=10,
        adaptive_tree_min_train_valid_split_improvement=0.01,
        adaptive_tree_overwrite_metric=None,
        adaptive_tree_skip_class_missing=True,
        adaptive_tree_test_size=None,
        average_logits=False,
    ):
        """Initialize a DecisionTreeTabPFN model."""
        # Convert numpy ints to Python ints to avoid issues with TabPFN client JSON serialization
        if hasattr(min_samples_split, "item"):
            min_samples_split = int(min_samples_split)
        if hasattr(min_samples_leaf, "item"):
            min_samples_leaf = int(min_samples_leaf)
        if max_depth is not None and hasattr(max_depth, "item"):
            max_depth = int(max_depth)
        if max_leaf_nodes is not None and hasattr(max_leaf_nodes, "item"):
            max_leaf_nodes = int(max_leaf_nodes)
        if random_state is not None and hasattr(random_state, "item"):
            random_state = int(random_state)

        # Optional args are arguments that may not be present in BaseDecisionTree,
        # depending on the sklearn version. Helps maintain compatibility.
        optional_args = {"monotonic_cst": monotonic_cst}
        filtered_optional_args = {
            k: v for k, v in optional_args.items() if v is not None
        }

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
        self.monotonic_cst = monotonic_cst
        self.ccp_alpha = ccp_alpha
        self.filtered_optional_args = filtered_optional_args

        self.X_valid = X_valid
        self.y_valid = y_valid
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
        self.adaptive_tree_patience = adaptive_tree_patience
        self.adaptive_tree_min_train_valid_split_improvement = (
            adaptive_tree_min_train_valid_split_improvement
        )
        self.adaptive_tree_overwrite_metric = adaptive_tree_overwrite_metric
        self.adaptive_tree_skip_class_missing = adaptive_tree_skip_class_missing
        self.adaptive_tree_test_size = adaptive_tree_test_size
        self.average_logits = average_logits

        self.categorical_features = categorical_features

    def fit(self, X: NDArray, y: NDArray, sample_weight=None, check_input: bool = True):
        """Fit the decision tree to the data.

        Args:
            X: Training data features
            y: Training data targets
            sample_weight: Sample weights
            check_input: Whether to validate the input

        Returns:
            Fitted estimator
        """
        return self._fit(X, y, sample_weight, check_input=check_input)

    def _fit(
        self,
        X: NDArray,
        y: NDArray,
        sample_weight=None,
        check_input: bool = True,
        missing_values_in_feature_mask=None,
    ):
        """Fit the DecisionTree-TabPFN model.

        Args:
            X: Training data
            y: Target values
            sample_weight: Sample weights for fitting
            check_input: Whether to validate input data
            missing_values_in_feature_mask: Mask for missing values (not used)

        Returns:
            Fitted estimator
        """
        # Generate a random seed if none was provided
        if self.tree_seed is None:
            self.tree_seed = random.randint(1, 10000)

        if check_input:
            # Input validation could be done here
            pass

        # Handle X and y
        self.X = X
        self.y = y
        self.n_features_in_ = X.shape[1]

        if isinstance(y, pd.Series):
            self.y = y.values

        if self.task_type == "multiclass":
            self.classes_ = np.unique(y)
            self.n_classes_ = len(self.classes_)
            self.train_label_encoder = LabelEncoder()
            self.train_y_transformed = self.train_label_encoder.fit_transform(self.y)
        elif self.task_type == "regression":
            self.train_y_transformed = self.y

        # Init decision tree
        self.tree_ = self.init_decision_tree()

        # If we're doing adaptive tree (selective node fitting),
        # we need validation data for pruning
        if self.adaptive_tree:
            # Skip adaptive tree if any class has only a single sample
            if (
                self.task_type == "multiclass"
                and len(np.unique(y)) > 1
                and np.unique(y, return_counts=True)[1].min() == 1
            ):
                # If a class has only one sample, we can't split properly
                self.adaptive_tree = False

            # Create validation split if not provided
            if self.X_valid is None or self.y_valid is None:
                # Min valid samples based on train set size
                min_valid_samples = max(
                    2,
                    int(
                        len(X)
                        * 0.2
                        * self.adaptive_tree_min_valid_samples_fraction_of_train,
                    ),
                )

                # Ensure adequate train and validation sizes
                if len(X) < self.adaptive_tree_min_train_samples + min_valid_samples:
                    self.adaptive_tree = False
                else:
                    # Create the train-validation split
                    (
                        train_X,
                        valid_X,
                        train_y,
                        valid_y,
                    ) = train_test_split(
                        X,
                        y,
                        test_size=min(
                            0.2,
                            min_valid_samples / len(X),
                        ),
                        random_state=self.random_state,
                        stratify=y if self.task_type == "multiclass" else None,
                    )

                    # Use the split for tree building
                    self.X = train_X
                    self.y = train_y
                    self.X_valid = valid_X
                    self.y_valid = valid_y

                    # Update transformed y if needed
                    if self.task_type == "multiclass":
                        self.train_y_transformed = self.train_label_encoder.transform(
                            self.y,
                        )
                    elif self.task_type == "regression":
                        self.train_y_transformed = self.y

        # Fit the tree to the data
        self.tree_.fit(
            self.X,
            self.train_y_transformed,
            sample_weight=sample_weight,
            check_input=check_input,
        )

        # Prepare data for leaf model training
        self.todo_post_fit = False
        self.fit_leafs(self.X, self.train_y_transformed)

        # Perform any post-fit operations
        self.post_fit()

        return self

    def fit_leafs(self, train_X: NDArray, train_y: NDArray) -> None:
        """Fit TabPFN models at each leaf of the decision tree.

        Args:
            train_X: Training data features
            train_y: Training data targets
        """
        self.leaf_train_data = {}
        self.leaf_tabpfn = {}

        # Get all leaf nodes in the tree
        leaf_node_ids = self.tree_.tree_.children_left == -1
        leaf_ids = np.where(leaf_node_ids)[0]

        # For each leaf node, collect the training samples that fall into it
        # We will then train a separate TabPFN model for each leaf
        train_indices_at_leaf = {}
        for _i, leaf_id in enumerate(leaf_ids):
            # Get samples that end up at this leaf
            leaf_samples_mask = (
                self.tree_.decision_path(train_X).toarray()[:, leaf_id] > 0
            )
            train_indices = np.where(leaf_samples_mask)[0]

            # Store for later use
            train_indices_at_leaf[leaf_id] = train_indices
            if len(train_indices) > 0:
                # Store the training data for this leaf
                self.leaf_train_data[leaf_id] = {
                    "X": train_X[train_indices],
                    "y": train_y[train_indices],
                }

    def get_tree(self) -> BaseDecisionTree:
        """Get the underlying decision tree.

        Returns:
            The sklearn decision tree object
        """
        return self.tree_

    def apply_tree(self, X: NDArray) -> NDArray:
        """Apply the decision tree to get decision paths for each sample.

        Args:
            X: Input data

        Returns:
            Decision paths indicating which nodes were traversed
        """
        # Preprocess data if needed
        X_preprocessed = self.preprocess_data_for_tree(X)

        # Get decision path matrix (samples x nodes)
        decision_path = self.tree_.decision_path(X_preprocessed)

        # Convert to dense matrix and expand dimensions for compatibility
        return np.expand_dims(decision_path.todense(), 2)

    def apply_tree_train(
        self,
        X: NDArray,
        y: NDArray,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Apply the tree to training data and return paths along with features and targets.

        Args:
            X: Input features
            y: Target values

        Returns:
            Tuple of (decision paths, features, targets)
        """
        return self.apply_tree(X), X, y

    def init_decision_tree(self) -> BaseDecisionTree:
        """Initialize the decision tree object.

        Returns:
            A newly created decision tree object
        """
        raise NotImplementedError("init_decision_tree must be implemented")

    def post_fit(self) -> None:
        """Perform any necessary operations after fitting the tree."""

    def preprocess_data_for_tree(self, X: NDArray) -> NDArray:
        """Preprocess data before applying the decision tree.

        Args:
            X: Input data

        Returns:
            Preprocessed data
        """
        if torch.is_tensor(X):
            X = X.cpu().numpy()

        return X

    def predict_(
        self,
        X: NDArray,
        y: NDArray | None = None,
        check_input: bool = True,
    ) -> NDArray:
        """Predict class probabilities for input data.

        Args:
            X: Data that should be evaluated
            y: Optional target values (not used in prediction)
            check_input: Whether to validate the input

        Returns:
            Probabilities of each class
        """
        # If post-fit operations are still needed, perform them now
        if self.todo_post_fit:
            self.post_fit()

            # Also verify the model using validation data if available
            if (
                self.adaptive_tree
                and self.X_valid is not None
                and self.y_valid is not None
            ):
                if isinstance(self.y_valid, pd.Series):
                    valid_y_values = self.y_valid.values
                else:
                    valid_y_values = self.y_valid

                self.predict_(
                    self.X_valid,
                    valid_y_values,
                    check_input=False,
                )

            self.todo_post_fit = False

        # Preprocess input data
        X_preprocessed = self.preprocess_data_for_tree(X)

        # Get leaf nodes for each data point
        # Matrix of shape (n_samples, n_nodes, n_estimators) with a one if the sample
        # went through that decision node
        X_test_leaf_nodes = self.apply_tree(X_preprocessed)
        N_samples, N_nodes, N_estimators = X_test_leaf_nodes.shape

        # Initialize prediction arrays
        y_prob = {}
        y_prob_averaging = self.init_eval_prob(N_samples)

        # Flag for adaptive tree pruning
        prune_leafs_and_nodes = y is not None and self.adaptive_tree
        if prune_leafs_and_nodes:
            # Track node prediction type (pruned or not)
            self.node_prediction_type = {}
            for estimator_id in range(N_estimators):
                self.node_prediction_type[estimator_id] = {}

        # Process each leaf node
        for estimator_id in range(N_estimators):
            y_prob[estimator_id] = {}
            leaf_node_ids = self.tree_.tree_.children_left == -1
            leaf_ids = np.where(leaf_node_ids)[0]

            # For each leaf node, get predictions
            for leaf_id in leaf_ids:
                # Skip first leaf of new tree after first tree
                # (this is always the same)
                if estimator_id > 0 and leaf_id == 0:
                    continue

                # Check which test samples ended up at this leaf
                leaf_mask = X_test_leaf_nodes[:, leaf_id, estimator_id] > 0
                test_sample_indices = np.where(leaf_mask)[0]

                if len(test_sample_indices) == 0:
                    # No test samples fell into this leaf
                    continue

                # Get training data for this leaf
                if leaf_id not in self.leaf_train_data:
                    # No training data for this leaf - use uniform prediction
                    leaf_prediction = self.init_eval_prob(N_samples)
                    y_prob[estimator_id][leaf_id] = leaf_prediction
                else:
                    X_train_samples = self.leaf_train_data[leaf_id]["X"]
                    y_train_samples = self.leaf_train_data[leaf_id]["y"]

                    # Use separate TabPFN model for each leaf
                    leaf_prediction = self.predict_leaf_(
                        X_train_samples,
                        y_train_samples,
                        leaf_id,
                        X_preprocessed[test_sample_indices],
                        test_sample_indices,
                    )

                    # Store predictions
                    y_prob[estimator_id][leaf_id] = leaf_prediction

                # Apply the leaf predictions to the test samples
                if self.task_type == "multiclass":
                    # For classification, we need to handle 2D predictions
                    # log(p) is needed for classifier
                    for i, idx in enumerate(test_sample_indices):
                        y_prob_averaging[idx] = np.log(leaf_prediction[i] + 1e-6)

                    # Convert logits to probabilities
                    for idx in test_sample_indices:
                        y_prob_averaging[idx] = DecisionTreeTabPFNBase.softmax(
                            y_prob_averaging[idx].reshape(1, -1),
                        ).reshape(-1)
                else:
                    # For regression, a simple assignment is enough (1D)
                    y_prob_averaging[test_sample_indices] = leaf_prediction

        # Return the final predictions
        return y_prob_averaging

    def get_optimize_metric(self) -> str:
        """Get the optimization metric to use for adaptive trees.

        Returns:
            The metric name to use
        """
        if self.adaptive_tree_overwrite_metric is not None:
            return self.adaptive_tree_overwrite_metric

        if self.task_type == "multiclass":
            return "logloss"
        if self.task_type == "regression":
            return "rmse"

        # Default if nothing else matches
        return "logloss"

    @staticmethod
    def softmax(logits: NDArray) -> NDArray:
        """Apply softmax function to convert logits to probabilities.

        Args:
            logits: Input logits

        Returns:
            Probabilities (values sum to 1 across classes)
        """
        exp_logits = np.exp(logits)  # Apply exponential to each logit
        sum_exp_logits = np.sum(
            exp_logits,
            axis=1,
            keepdims=True,
        )  # Sum across classes
        return exp_logits / sum_exp_logits  # Normalize by sum

    def predict_leaf_(
        self,
        X_train_samples: NDArray,
        y_train_samples: NDArray,
        leaf_id: int,
        X: NDArray,
        indices: NDArray,
    ) -> NDArray:
        """Predict using the TabPFN model for a specific leaf.

        Args:
            X_train_samples: Training data for this leaf
            y_train_samples: Training targets for this leaf
            leaf_id: ID of the leaf node
            X: Test data to predict
            indices: Indices of test samples

        Returns:
            Predicted probabilities
        """
        raise NotImplementedError("predict_leaf_ must be implemented")

    def init_eval_prob(self, n_samples: int, to_zero: bool = False) -> NDArray:
        """Initialize evaluation probability array.

        Args:
            n_samples: Number of samples
            to_zero: Whether to initialize to zeros instead of uniform probabilities

        Returns:
            Initialized probability array
        """
        if self.task_type == "multiclass":
            if to_zero:
                y_eval_prob = np.zeros((n_samples, self.n_classes_))
            else:
                y_eval_prob = np.ones((n_samples, self.n_classes_)) / self.n_classes_
        elif self.task_type == "regression":
            y_eval_prob = np.zeros(n_samples)

        return y_eval_prob


class DecisionTreeTabPFNClassifier(DecisionTreeTabPFNBase, ClassifierMixin):
    """TabPFN-enhanced decision tree for classification.

    This class combines a decision tree with TabPFN classifier models at
    the leaves, enhancing prediction performance.
    """

    _estimator_type = "classifier"

    task_type = "multiclass"

    def predict_leaf_(
        self,
        X_train_samples: NDArray,
        y_train_samples: NDArray,
        leaf_id: int,
        X: NDArray,
        indices: NDArray,
    ) -> NDArray:
        """Predict class probabilities using TabPFN for a specific leaf.

        Args:
            X_train_samples: Training data for this leaf
            y_train_samples: Training targets for this leaf
            leaf_id: ID of the leaf node
            X: Test samples to predict
            indices: Indices of test samples

        Returns:
            Class probabilities
        """
        # Ensure we use standard Python int (not numpy.int64) to avoid serialization issues
        # with TabPFN client when it tries to convert the value to JSON
        leaf_id_int = int(leaf_id) if hasattr(leaf_id, "item") else leaf_id
        tree_seed_int = (
            int(self.tree_seed) if hasattr(self.tree_seed, "item") else self.tree_seed
        )

        y_eval_prob = self.init_eval_prob(X.shape[0], to_zero=True)
        classes = np.array(np.unique(list(map(int, y_train_samples))))

        if len(classes) == 1:
            # Only one class in training data - predict that class with 100% probability
            y_eval_prob[:, classes[0]] = 1.0
            return y_eval_prob

        # Fit TabPFN to this leaf's training data
        try:
            # Set a seed based on leaf ID for reproducibility (using Python ints, not NumPy ints)
            # This addition will return a Python int even if both inputs are Python ints
            self.tabpfn.random_state = leaf_id_int + tree_seed_int
            self.tabpfn.fit(X_train_samples, y_train_samples)

            # Get predictions for test samples
            pred_proba = self.tabpfn.predict_proba(X)
            y_eval_prob[:, classes] = pred_proba
        except Exception as e:
            # Handle specific TabPFN errors gracefully
            if (
                not isinstance(e, ValueError)
                or (len(e.args) == 0)
                or e.args[0]
                != "All features are constant and would have been removed! Unable to predict using TabPFN."
            ):
                raise e

            # If all features are constant, use class distribution from training
            warnings.warn(
                "All features in leaf are constant or otherwise not suitable to predict with TabPFN. Returning class distribution as predictions.",
                UserWarning,
                stacklevel=2,
            )

            # Get class distribution from training data
            counts = np.bincount(y_train_samples.astype(int), minlength=self.n_classes_)
            probs = counts / counts.sum()

            # Apply to all test samples
            for j in range(len(y_eval_prob)):
                y_eval_prob[j, classes] = probs[classes]

        return y_eval_prob

    def predict(self, X: NDArray, check_input: bool = True) -> NDArray:
        """Predict class labels for input data.

        Args:
            X: Data to evaluate
            check_input: Whether to validate input

        Returns:
            Predicted class labels
        """
        from sklearn.utils.validation import check_is_fitted

        check_is_fitted(self, ["tree_", "classes_"])

        return np.argmax(
            self.predict_proba(X, check_input),
            axis=1,
        )

    def predict_proba(self, X: NDArray, check_input: bool = True) -> NDArray:
        """Predict class probabilities for input data.

        Args:
            X: Data to evaluate
            check_input: Whether to validate input

        Returns:
            Class probabilities
        """
        from sklearn.utils.validation import check_is_fitted

        check_is_fitted(self, ["tree_", "classes_"])

        return self.predict_(X, check_input=check_input)

    def post_fit(self) -> None:
        """Perform operations after fitting the tree."""

    def init_decision_tree(self) -> DecisionTreeClassifier:
        """Initialize the decision tree classifier.

        Returns:
            New DecisionTreeClassifier instance
        """
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
            ccp_alpha=self.ccp_alpha,
            **self.filtered_optional_args,
        )


class DecisionTreeTabPFNRegressor(DecisionTreeTabPFNBase, RegressorMixin):
    """TabPFN-enhanced decision tree for regression.

    This class combines a decision tree with TabPFN regressor models at
    the leaves, enhancing prediction performance for regression tasks.
    """

    _estimator_type = "regressor"

    task_type = "regression"

    def __init__(
        self,
        *,
        tabpfn,
        criterion="squared_error",  # Use regression-appropriate criterion
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        ccp_alpha=0.0,
        monotonic_cst=None,
        categorical_features=None,
        X_valid=None,
        y_valid=None,
        verbose=0,
        show_progress=True,
        fit_nodes=False,
        tree_seed=None,
        adaptive_tree=False,
        adaptive_tree_min_train_samples=10,
        adaptive_tree_max_train_samples=10000,
        adaptive_tree_min_valid_samples_fraction_of_train=0.5,
        adaptive_tree_patience=10,
        adaptive_tree_min_train_valid_split_improvement=0.01,
        adaptive_tree_overwrite_metric=None,
        adaptive_tree_skip_class_missing=True,
        adaptive_tree_test_size=None,
        average_logits=False,
    ):
        """Initialize a regression tree with TabPFN models at the leaves.

        For parameter descriptions, see the base class.
        """
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
            X_valid=X_valid,
            y_valid=y_valid,
            verbose=verbose,
            show_progress=show_progress,
            fit_nodes=fit_nodes,
            tree_seed=tree_seed,
            adaptive_tree=adaptive_tree,
            adaptive_tree_min_train_samples=adaptive_tree_min_train_samples,
            adaptive_tree_max_train_samples=adaptive_tree_max_train_samples,
            adaptive_tree_min_valid_samples_fraction_of_train=adaptive_tree_min_valid_samples_fraction_of_train,
            adaptive_tree_patience=adaptive_tree_patience,
            adaptive_tree_min_train_valid_split_improvement=adaptive_tree_min_train_valid_split_improvement,
            adaptive_tree_overwrite_metric=adaptive_tree_overwrite_metric,
            adaptive_tree_skip_class_missing=adaptive_tree_skip_class_missing,
            adaptive_tree_test_size=adaptive_tree_test_size,
            average_logits=average_logits,
        )

    def _get_tags(self) -> dict[str, Any]:
        """Get scikit-learn compatible estimator tags.

        Returns:
            Dictionary of estimator tags
        """
        # Get parent tags
        tags = super()._get_tags()

        # Add regression-specific tags
        tags.update({"requires_y": True})

        return tags

    def init_decision_tree(self) -> DecisionTreeRegressor:
        """Initialize the decision tree regressor.

        Returns:
            New DecisionTreeRegressor instance
        """
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
            **self.filtered_optional_args,
        )

    def predict_leaf_(
        self,
        X_train_samples: NDArray,
        y_train_samples: NDArray,
        leaf_id: int,
        X: NDArray,
        indices: NDArray,
    ) -> NDArray:
        """Predict regression values using TabPFN for a specific leaf.

        Args:
            X_train_samples: Training data for this leaf
            y_train_samples: Training targets for this leaf
            leaf_id: ID of the leaf node
            X: Test samples to predict
            indices: Indices of test samples

        Returns:
            Predicted regression values
        """
        # Ensure we use standard Python int (not numpy.int64) to avoid serialization issues
        # with TabPFN client when it tries to convert the value to JSON
        leaf_id_int = int(leaf_id) if hasattr(leaf_id, "item") else leaf_id
        tree_seed_int = (
            int(self.tree_seed) if hasattr(self.tree_seed, "item") else self.tree_seed
        )

        # Initialize prediction array
        y_pred = np.zeros(X.shape[0])

        # TabPFN requires at least 2 samples for training
        if len(X_train_samples) < 2:
            # For single sample case, just use the target value directly
            # This is a reasonable fallback for regression
            warnings.warn(
                f"Leaf node {leaf_id} has only {len(X_train_samples)} samples, which is insufficient for TabPFN (needs at least 2). "
                "Using the mean target value for predictions.",
                UserWarning,
                stacklevel=2,
            )
            # Fill predictions with the mean target value from this leaf
            y_pred.fill(np.mean(y_train_samples))
            return y_pred

        # Set random state (using Python ints, not NumPy ints)
        self.tabpfn.random_state = leaf_id_int + tree_seed_int

        try:
            # Fit TabPFN on the leaf data (now we know there are at least 2 samples)
            self.tabpfn.fit(X_train_samples, y_train_samples)
            # Get predictions
            return self.tabpfn.predict(X)
        except (ValueError, RuntimeError, NotImplementedError) as e:
            # Handle specific TabPFN errors gracefully
            warnings.warn(
                f"TabPFN prediction failed for leaf {leaf_id}: {str(e)}. "
                "Using the mean target value for predictions.",
                UserWarning,
                stacklevel=2,
            )
            # Fallback to using the mean target value
            y_pred.fill(np.mean(y_train_samples))
            return y_pred

    def predict(self, X: NDArray, check_input: bool = True) -> NDArray:
        """Predict regression values for input data.

        Args:
            X: Data to evaluate
            check_input: Whether to validate input

        Returns:
            Predicted regression values
        """
        from sklearn.utils.validation import check_is_fitted

        check_is_fitted(self, ["tree_"])

        return self.predict_(X, check_input=False)

    def post_fit(self) -> None:
        """Perform operations after fitting the tree."""
