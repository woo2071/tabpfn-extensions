#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

"""ClassifierAsRegressor: Converting TabPFN classifiers to regressors.

This module provides a wrapper class that adapts TabPFN classifiers to solve
regression problems by discretizing the target variable into bins and treating
the problem as classification. It then converts the predicted probabilities
back to continuous values.

The main class is ClassifierAsRegressor, which follows scikit-learn's API
and can be used as a drop-in replacement for TabPFNRegressor when only
the classifier implementation is available (e.g., when using TabPFN client).

Key features:
- Converts continuous targets to discrete classes
- Maps class probabilities back to expected values
- Compatible with both TabPFN and TabPFN-client backends
- Follows scikit-learn Estimator interface

Example usage:
    ```python
    from tabpfn import TabPFNClassifier  # or from tabpfn_client
    from tabpfn_extensions.classifier_as_regressor import ClassifierAsRegressor

    # Create a TabPFN classifier
    clf = TabPFNClassifier()

    # Wrap it with ClassifierAsRegressor
    reg = ClassifierAsRegressor(clf, n_bins=10)

    # Use it like any scikit-learn regressor
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    ```
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import LabelEncoder

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Try to import TabPFN's BarDistribution class, or create a simplified version
# if not available
try:
    from tabpfn.model.bar_distribution import BarDistribution
except ImportError:
    # Fallback implementation for TabPFN 2.x or TabPFN client
    class BarDistributionImpl:
        """Simplified version of TabPFN's BarDistribution for compatibility.

        This class provides minimal functionality needed by ClassifierAsRegressor
        when the full TabPFN implementation is not available.

        Parameters
        ----------
        borders : torch.Tensor
            The bin borders for discretization.
        """

        def __init__(self, borders: torch.Tensor) -> None:
            """Initialize the BarDistribution with bin borders.

            Args:
                borders: Tensor containing the borders between bins
            """
            self.borders = borders

    # Use alias to avoid name conflict
    BarDistribution = BarDistributionImpl


class ClassifierAsRegressor(RegressorMixin, BaseEstimator):
    """Wrapper class to use a classifier as a regressor.

    This class takes a classifier estimator and converts it into a regressor by
    encoding the target labels and treating the regression problem as a
    classification task.

    Parameters:
        estimator : BaseEstimator
            Classifier estimator to be used as a regressor.

    Attributes:
        label_encoder_ : LabelEncoder
            Label encoder used to transform target regression labels to classes.
        y_train_ : array-like of shape (n_samples,)
            Transformed target labels used for training.
        categorical_features : list
            List of categorical feature indices.

    Examples:
    ```python title="Example"
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.model_selection import train_test_split
    >>> from tabpfn_extensions import ManyClassClassifier, TabPFNClassifier
    >>> from tabpfn_extensions import ClassifierAsRegressor
    >>> x, y = load_diabetes(return_X_y=True)
    >>> x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)
    >>> clf = TabPFNClassifier()
    >>> # Use ManyClassClassifier for regression targets requiring many bins
    >>> clf = ManyClassClassifier(clf, n_estimators=10, alphabet_size=10)
    >>> reg = ClassifierAsRegressor(clf)
    >>> reg.fit(x_train, y_train)
    >>> y_pred = reg.predict(x_test)
    ```
    """

    def __init__(self, estimator: BaseEstimator) -> None:
        """Initialize the ClassifierAsRegressor.

        Args:
            estimator: A classifier estimator with a predict_proba method

        Raises:
            AssertionError: If estimator doesn't have a predict_proba method
        """
        assert hasattr(estimator, "predict_proba"), (
            "The estimator must have a predict_proba method to be used as a regressor."
        )
        self.estimator = estimator
        self.categorical_features: list[int] | None = None

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> ClassifierAsRegressor:
        """Fit the classifier as a regressor.

        Parameters:
            X : array-like of shape (n_samples, n_features)
                Training data.
            y : array-like of shape (n_samples,)
                Target labels.

        Returns:
            self : ClassifierAsRegressor
                Fitted estimator.
        """
        self.label_encoder_ = LabelEncoder()
        y_transformed = self.label_encoder_.fit_transform(y)
        self.y_train_ = y_transformed

        self.estimator.fit(X, y_transformed)
        return self

    def _get_bar_dist(self) -> BarDistribution:
        """Get the bar distribution for the target labels.

        Returns:
            BarDistribution
                Bar distribution for the target labels.
        """
        unique_y = self.label_encoder_.classes_
        bucket_widths = unique_y[1:] - unique_y[:-1]

        borders = [
            *(
                unique_y - np.array([bucket_widths[0], *bucket_widths.tolist()]) / 2
            ).tolist(),
            unique_y[-1] + bucket_widths[-1] / 2,
        ]
        return BarDistribution(borders=torch.tensor(borders).float())

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict the target values for the input data.

        Parameters:
            X : array-like of shape (n_samples, n_features)
                Input data.

        Returns:
            y_pred : array-like of shape (n_samples,)
                Predicted target values.
        """
        try:
            return self.predict_full(X)["mean"]
        except (ValueError, KeyError, IndexError, RuntimeError, AttributeError):
            # Fallback method that just uses the most likely class and converts back
            proba = self.estimator.predict_proba(X)
            pred_class = np.argmax(proba, axis=1)
            return self.label_encoder_.inverse_transform(pred_class)

    def set_categorical_features(self, categorical_features: list[int]) -> None:
        """Set the categorical feature indices.

        Parameters:
            categorical_features : list
                List of categorical feature indices.
        """
        self.categorical_features = categorical_features

    def get_optimization_mode(self) -> str:
        """Get the optimization mode for the regressor.

        Returns:
            str
                Optimization mode ("mean").
        """
        return "mean"

    @staticmethod
    def probabilities_to_logits_multiclass(
        probabilities: NDArray[np.float64],
        eps: float = 1e-6,
    ) -> NDArray[np.float64]:
        """Convert probabilities to logits for a multi-class problem.

        Parameters:
            probabilities : array-like of shape (n_samples, n_classes)
                Input probabilities for each class.
            eps : float, default=1e-6
                Small value to avoid division by zero or taking logarithm of zero.

        Returns:
            logits : array-like of shape (n_samples, n_classes)
                Output logits for each class.
        """
        probabilities = np.clip(probabilities, eps, 1 - eps)
        remaining_prob = 1 - probabilities.sum(axis=1, keepdims=True)
        remaining_prob = np.clip(remaining_prob, eps, 1 - eps)
        return np.log(probabilities) - np.log(remaining_prob)

    def predict_full(self, X: NDArray[np.float64]) -> dict[str, Any]:
        """Predict the full set of output values for the input data.

        Parameters:
            X : array-like of shape (n_samples, n_features)
                Input data.

        Returns:
            dict
                Dictionary containing the predicted output values, including:
                - "mean": Predicted mean values.
                - "median": Predicted median values.
                - "mode": Predicted mode values.
                - "logits": Predicted logits.
                - "buckets": Predicted bucket probabilities.
        """
        class_probs = self.estimator.predict_proba(X)

        # Get the class values from label encoder
        unique_classes = self.label_encoder_.classes_

        # Calculate mean prediction using class probabilities
        weighted_mean = np.sum(class_probs * unique_classes.reshape(1, -1), axis=1)

        # Calculate median (approximation)
        cumulative_probs = np.cumsum(class_probs, axis=1)
        median_indices = np.argmax(cumulative_probs >= 0.5, axis=1)
        median_values = unique_classes[median_indices]

        # Calculate mode (class with highest probability)
        mode_indices = np.argmax(class_probs, axis=1)
        mode_values = unique_classes[mode_indices]

        # Convert probabilities to logits for compatibility
        class_logits = torch.tensor(
            ClassifierAsRegressor.probabilities_to_logits_multiclass(class_probs),
        ).float()

        # Get the bar distribution
        self._get_bar_dist()

        # Return results in a format similar to TabPFNRegressor's output
        return {
            "mean": weighted_mean,
            "median": median_values,
            "mode": mode_values,
            "logits": class_logits,
            "buckets": class_probs,
        }
