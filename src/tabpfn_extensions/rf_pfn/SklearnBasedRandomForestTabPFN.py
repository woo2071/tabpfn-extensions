"""TODO: Don't copy data but keep indices instead."""

#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

from __future__ import annotations

import logging
import time

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from .SklearnBasedDecisionTreeTabPFN import (
    DecisionTreeTabPFNClassifier,
    DecisionTreeTabPFNRegressor,
)

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("RF-PFN")


def softmax_numpy(logits):
    exp_logits = np.exp(logits)  # Apply exponential to each logit
    sum_exp_logits = np.sum(
        exp_logits,
        axis=-1,
        keepdims=True,
    )  # Sum of exponentials across classes
    return exp_logits / sum_exp_logits  # Normalize to get probabilities


class RandomForestTabPFNBase:
    """Base Class for common functionalities."""

    def get_n_estimators(self, X):
        return self.n_estimators

    def fit(self, X, y, sample_weight=None):
        """Fits RandomForestTabPFN
        :param X: Feature training data
        :param y: Label training data
        :param sample_weight: Weights of each sample
        :return: None.
        """
        self.estimator = self.init_base_estimator()
        self.X = X
        self.n_estimators = self.get_n_estimators(X)
        time.time()

        # Convert tensors to numpy if needed
        if torch.is_tensor(X):
            X = X.numpy()
        if torch.is_tensor(y):
            y = y.numpy()

        # Special case for depth 0 - just use TabPFN directly
        if self.max_depth == 0:
            self.tabpfn.fit(X, y)
            return self

        # Initialize the tree estimators
        n_estimators = self.n_estimators
        if n_estimators <= 0:
            raise ValueError(
                f"n_estimators must be greater than zero, got {n_estimators}"
            )

        # Initialize estimators list
        self.estimators_ = []

        # Generate bootstrapped datasets and fit trees
        for i in range(n_estimators):
            # Clone the base estimator
            tree = self.init_base_estimator()

            # Bootstrap sample if requested (like in RandomForest)
            if self.bootstrap:
                n_samples = X.shape[0]
                indices = np.random.choice(
                    n_samples,
                    size=n_samples
                    if self.max_samples is None
                    else int(self.max_samples * n_samples),
                    replace=True,
                )
                X_boot = X[indices]
                y_boot = y[indices]
            else:
                X_boot = X
                y_boot = y

            # Fit the tree on bootstrapped data
            tree.fit(X_boot, y_boot)
            self.estimators_.append(tree)

        # Track features seen during fit
        self.n_features_in_ = X.shape[1]

        # Set flag to indicate successful fit
        self._fitted = True

        return self


class RandomForestTabPFNClassifier(RandomForestTabPFNBase, RandomForestClassifier):
    """RandomForestTabPFNClassifier."""

    task_type = "multiclass"

    def __init__(
        self,
        tabpfn=None,
        n_jobs=1,
        categorical_features=None,
        show_progress=False,
        verbose=0,
        adaptive_tree=True,
        fit_nodes=True,
        adaptive_tree_overwrite_metric="log_loss",
        adaptive_tree_test_size=0.2,
        adaptive_tree_min_train_samples=100,
        adaptive_tree_max_train_samples=5000,
        adaptive_tree_min_valid_samples_fraction_of_train=0.2,
        preprocess_X_once=False,
        max_predict_time=60,
        rf_average_logits=True,
        dt_average_logits=True,
        adaptive_tree_skip_class_missing=True,
        # Added to make cloneable.
        n_estimators=100,
        criterion="gini",
        max_depth=5,
        min_samples_split=1000,
        min_samples_leaf=5,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        random_state=None,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
    ):
        """:param N_ensemble_configurations: Parameter of TabPFNClassifier
        :param n_estimators: Number of DT-TabPFN models
        :param min_samples_split: min_samples_split param of DTTabPFN
        :param max_features: sklearn DT param
        :param bootstrap: Whether use bootstrapping or not
        :param n_jobs: parallel processing of submodels n_jobs will be processed simultaniously
        :param random_state: random state of sklearn RF model
        """
        super().__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            verbose=verbose,
            n_jobs=n_jobs,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            random_state=random_state,
            warm_start=warm_start,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
        )

        self.tabpfn = tabpfn

        self.categorical_features = categorical_features
        self.show_progress = show_progress
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.adaptive_tree = adaptive_tree
        self.fit_nodes = fit_nodes
        self.adaptive_tree_overwrite_metric = adaptive_tree_overwrite_metric
        self.adaptive_tree_test_size = adaptive_tree_test_size
        self.adaptive_tree_min_train_samples = adaptive_tree_min_train_samples
        self.adaptive_tree_max_train_samples = adaptive_tree_max_train_samples
        self.adaptive_tree_min_valid_samples_fraction_of_train = (
            adaptive_tree_min_valid_samples_fraction_of_train
        )
        self.preprocess_X_once = preprocess_X_once
        self.max_predict_time = max_predict_time
        self.rf_average_logits = rf_average_logits
        self.dt_average_logits = dt_average_logits
        self.adaptive_tree_skip_class_missing = adaptive_tree_skip_class_missing
        self.n_estimators = n_estimators

    def __sklearn_tags__(self):
        # Create a dictionary of tags directly (without relying on parent __sklearn_tags__)
        # This avoids issues with sklearn's internal estimation of "allow_nan" which
        # tries to create a new base estimator without tabpfn
        tags = {
            "allow_nan": True,
            "estimator_type": "classifier",
            "poor_score": False,
            "no_validation": False,
            "multioutput": False,
            "requires_positive_X": False,
            "requires_positive_y": False,
            "X_types": ["2darray"],
            "preserves_dtype": [],
        }
        return tags

    def init_base_estimator(self):
        return DecisionTreeTabPFNClassifier(
            tabpfn=self.tabpfn,
            min_samples_split=self.min_samples_split,
            max_features=self.max_features,
            random_state=self.random_state,
            categorical_features=self.categorical_features,
            max_depth=self.max_depth,
            show_progress=self.show_progress,
            adaptive_tree=self.adaptive_tree,
            fit_nodes=self.fit_nodes,
            verbose=self.verbose,
            adaptive_tree_test_size=self.adaptive_tree_test_size,
            adaptive_tree_overwrite_metric=self.adaptive_tree_overwrite_metric,
            adaptive_tree_min_train_samples=self.adaptive_tree_min_train_samples,
            adaptive_tree_max_train_samples=self.adaptive_tree_max_train_samples,
            adaptive_tree_min_valid_samples_fraction_of_train=self.adaptive_tree_min_valid_samples_fraction_of_train,
            average_logits=self.dt_average_logits,
            adaptive_tree_skip_class_missing=self.adaptive_tree_skip_class_missing,
        )

    def predict(self, X):
        """Predict class for X.

        The predicted class of an input sample is a vote by the trees in
        the forest, weighted by their probability estimates. That is,
        the predicted class is the one with highest mean probability
        estimate across the trees.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        Returns:
        -------
        y : ndarray of shape (n_samples,)
            The predicted classes.
        """
        # Get class probabilities
        proba = self.predict_proba(X)

        # Return class with highest probability
        if hasattr(self, "classes_"):
            return self.classes_.take(np.argmax(proba, axis=1), axis=0)
        else:
            return np.argmax(proba, axis=1)

    def _final_proba(self, all_proba, evaluated_estimators):
        for proba in all_proba:
            proba /= evaluated_estimators

        if self.rf_average_logits:
            # convert all_proba logits for the multiclass output to probabilities per class
            all_proba = softmax_numpy(all_proba)

        if len(all_proba) == 1:
            return all_proba[0]
        return all_proba

    def predict_proba(self, X):
        """Predict class probabilities for X.

        The predicted class probabilities of an input sample are computed as
        the mean predicted class probabilities of the trees in the forest.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        Returns:
        -------
        p : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples.
        """
        # Check if fitted
        if not hasattr(self, "_fitted") or not self._fitted:
            raise ValueError(
                "This RandomForestTabPFNClassifier instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator."
            )

        # Convert input if needed
        if torch.is_tensor(X):
            X = X.numpy()

        # Special case for depth 0
        if self.max_depth == 0:
            return self.tabpfn.predict_proba(X)

        # Get number of classes from first estimator
        if not hasattr(self, "n_classes_") and len(self.estimators_) > 0:
            self.n_classes_ = len(np.unique(self.estimators_[0].classes_))
            self.classes_ = self.estimators_[0].classes_

        # Initialize probabilities array
        n_samples = X.shape[0]
        all_proba = np.zeros((n_samples, self.n_classes_), dtype=np.float64)

        # Accumulate predictions from trees
        start_time = time.time()
        evaluated_estimators = 0

        for estimator in self.estimators_:
            # Get predictions from this tree
            proba = estimator.predict_proba(X)

            # Convert to logits if needed
            if self.rf_average_logits:
                proba = np.log(proba + 1e-10)  # Add small constant to avoid log(0)

            # Accumulate
            all_proba += proba

            # Check timeout
            evaluated_estimators += 1
            time_elapsed = time.time() - start_time
            if time_elapsed > self.max_predict_time and self.max_predict_time > 0:
                break

        # Average probabilities
        all_proba /= evaluated_estimators

        # Convert back from logits if needed
        if self.rf_average_logits:
            all_proba = softmax_numpy(all_proba)

        return all_proba


class RandomForestTabPFNRegressor(RandomForestTabPFNBase, RandomForestRegressor):
    """RandomForestTabPFNRegressor."""

    task_type = "regression"

    def __sklearn_tags__(self):
        # Create a dictionary of tags directly (without relying on parent __sklearn_tags__)
        # This avoids issues with sklearn's internal estimation of "allow_nan" which
        # tries to create a new base estimator without tabpfn
        tags = {
            "allow_nan": True,
            "estimator_type": "regressor",
            "poor_score": False,
            "no_validation": False,
            "multioutput": False,
            "requires_positive_X": False,
            "requires_positive_y": False,
            "X_types": ["2darray"],
            "preserves_dtype": [],
        }
        return tags

    def __init__(
        self,
        tabpfn=None,
        n_jobs=1,
        categorical_features=None,
        show_progress=False,
        verbose=0,
        adaptive_tree=True,
        fit_nodes=True,
        adaptive_tree_overwrite_metric="rmse",
        adaptive_tree_test_size=0.2,
        adaptive_tree_min_train_samples=100,
        adaptive_tree_max_train_samples=5000,
        adaptive_tree_min_valid_samples_fraction_of_train=0.2,
        preprocess_X_once=False,
        max_predict_time=-1,
        rf_average_logits=False,
        # Added to make cloneable.
        n_estimators=16,
        criterion="friedman_mse",
        max_depth=5,
        min_samples_split=300,
        min_samples_leaf=5,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        random_state=None,
        warm_start=False,
        ccp_alpha=0.0,
        max_samples=None,
    ):
        """:param N_ensemble_configurations: Parameter of TabPFNClassifier
        :param n_estimators: Number of DT-TabPFN models
        :param min_samples_split: min_samples_split param of DTTabPFN
        :param max_features: sklearn DT param
        :param bootstrap: Whether use bootstrapping or not
        :param n_jobs: parallel processing of submodels n_jobs will be processed simultaniously
        :param random_state: random state of sklearn RF model
        """
        super().__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
        )

        self.tabpfn = tabpfn

        self.categorical_features = categorical_features
        self.show_progress = show_progress
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.adaptive_tree = adaptive_tree
        self.fit_nodes = fit_nodes
        self.adaptive_tree_overwrite_metric = adaptive_tree_overwrite_metric
        self.adaptive_tree_test_size = adaptive_tree_test_size
        self.adaptive_tree_min_train_samples = adaptive_tree_min_train_samples
        self.adaptive_tree_max_train_samples = adaptive_tree_max_train_samples
        self.adaptive_tree_min_valid_samples_fraction_of_train = (
            adaptive_tree_min_valid_samples_fraction_of_train
        )
        self.preprocess_X_once = preprocess_X_once
        self.max_predict_time = max_predict_time
        self.rf_average_logits = rf_average_logits

    def init_base_estimator(self):
        return DecisionTreeTabPFNRegressor(
            tabpfn=self.tabpfn,
            min_samples_split=self.min_samples_split,
            max_features=self.max_features,
            random_state=self.random_state,
            categorical_features=self.categorical_features,
            max_depth=self.max_depth,
            show_progress=self.show_progress,
            adaptive_tree=self.adaptive_tree,
            fit_nodes=self.fit_nodes,
            verbose=self.verbose,
            adaptive_tree_test_size=self.adaptive_tree_test_size,
            adaptive_tree_overwrite_metric=self.adaptive_tree_overwrite_metric,
            adaptive_tree_min_train_samples=self.adaptive_tree_min_train_samples,
            adaptive_tree_max_train_samples=self.adaptive_tree_max_train_samples,
            adaptive_tree_min_valid_samples_fraction_of_train=self.adaptive_tree_min_valid_samples_fraction_of_train,
        )

    def predict(self, X):
        """Predict regression target for X.

        The predicted regression target of an input sample is computed as the
        mean predicted regression targets of the trees in the forest.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        Returns:
        -------
        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The predicted values.
        """
        # Check if fitted
        if not hasattr(self, "_fitted") or not self._fitted:
            raise ValueError(
                "This RandomForestTabPFNRegressor instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator."
            )

        # Convert input if needed
        if torch.is_tensor(X):
            X = X.numpy()

        # Special case for depth 0
        if self.max_depth == 0:
            return self.tabpfn.predict(X)

        # Initialize output array
        n_samples = X.shape[0]
        self.n_outputs_ = 1  # Only supporting single output for now
        y_hat = np.zeros(n_samples, dtype=np.float64)

        # Accumulate predictions from trees
        start_time = time.time()
        evaluated_estimators = 0

        for estimator in self.estimators_:
            # Get predictions from this tree
            pred = estimator.predict(X)

            # Accumulate
            y_hat += pred

            # Check timeout
            evaluated_estimators += 1
            time_elapsed = time.time() - start_time
            if time_elapsed > self.max_predict_time and self.max_predict_time > 0:
                break

        # Average predictions
        y_hat /= evaluated_estimators

        return y_hat


def _accumulate_prediction(predict, X, out, accumulate_logits=False):
    """This is a utility function for joblib's Parallel.

    It can't go locally in ForestClassifier or ForestRegressor, because joblib
    complains that it cannot pickle it when placed there.
    """
    prediction = predict(X, check_input=False)

    if accumulate_logits:
        # convert multiclass probabilities to logits
        prediction = np.log(prediction + 1e-10)  # Add small value to avoid log(0)

    if len(out) == 1:
        out[0] += prediction
    else:
        for i in range(len(out)):
            out[i] += prediction[i]
