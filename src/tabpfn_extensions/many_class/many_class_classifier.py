#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

"""ManyClassClassifier: TabPFN extension for handling classification with many classes.

This module provides a classifier that overcomes TabPFN's limitation on the number of
classes (typically 10) by using a meta-classifier approach. It works by breaking down
multi-class problems into multiple sub-problems, each within TabPFN's class limit.

Key features:
- Handles any number of classes beyond TabPFN's native limits
- Uses an efficient codebook approach to minimize the number of base models
- Compatible with both TabPFN and TabPFN-client backends
- Maintains high accuracy through redundant coding
- Follows scikit-learn's Estimator interface

Example usage:
    ```python
    from tabpfn import TabPFNClassifier  # or from tabpfn_client
    from tabpfn_extensions.many_class import ManyClassClassifier

    # Create a base TabPFN classifier
    base_clf = TabPFNClassifier()

    # Wrap it with ManyClassClassifier to handle more classes
    many_class_clf = ManyClassClassifier(
        estimator=base_clf,
        alphabet_size=10  # Use TabPFN's maximum class limit
    )

    # Use like any scikit-learn classifier, even with more than 10 classes
    many_class_clf.fit(X_train, y_train)
    y_pred = many_class_clf.predict(X_test)
    ```
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import (
    _num_samples,
    check_array,
    check_is_fitted,
)


class ManyClassClassifier(BaseEstimator, ClassifierMixin):
    """Output-Code multiclass strategy to extend TabPFN beyond its class limit.

    This class enables TabPFN to handle classification problems with any number of
    classes by using a meta-classifier approach. It creates an efficient coding
    system that maps the original classes to multiple sub-problems, each within
    TabPFN's class limit.

    Parameters
    ----------
    estimator : estimator object
        A classifier implementing fit() and predict_proba() methods.
        Typically a TabPFNClassifier instance. The base classifier
        should be able to handle up to `alphabet_size` classes.

    alphabet_size : int, default=None
        Maximum number of classes the base estimator can handle.
        If None, it will try to get this from estimator.max_num_classes_

    n_estimators : int, default=None
        Number of base estimators to train. If None, an optimal number
        is calculated based on the number of classes and alphabet_size.

    n_estimators_redundancy : int, default=4
        Redundancy factor for the auto-calculated number of estimators.
        Higher values increase reliability but also computational cost.

    random_state : int, RandomState instance, default=None
        Controls randomization used to initialize the codebook.
        Pass an int for reproducible results.

    Attributes:
    ----------
    classes_ : ndarray of shape (n_classes,)
        Array containing unique target labels.

    code_book_ : ndarray of shape (n_estimators, n_classes)
        N-ary array containing the coding scheme that maps original
        classes to base classifier problems.

    no_mapping_needed_ : bool
        True if the number of classes is within the alphabet_size limit,
        allowing direct use of the base estimator without any mapping.

    classes_index_ : dict
        Maps class labels to their indices in classes_.

    X_train : array-like
        Training data stored for reuse during prediction.

    Y_train : array-like
        Encoded training labels for each base estimator.

    Examples:
    --------
    >>> from sklearn.datasets import load_iris
    >>> from tabpfn import TabPFNClassifier
    >>> from tabpfn_extensions.many_class import ManyClassClassifier
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = load_iris(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    >>> base_clf = TabPFNClassifier()
    >>> many_clf = ManyClassClassifier(base_clf, alphabet_size=base_clf.max_num_classes_)
    >>> many_clf.fit(X_train, y_train)
    >>> y_pred = many_clf.predict(X_test)
    """

    def __init__(
        self,
        estimator,
        *,
        alphabet_size=None,
        n_estimators=None,
        n_estimators_redundancy=4,
        random_state=None,
    ):
        self.estimator = estimator
        self.random_state = random_state
        self.alphabet_size = alphabet_size
        self.n_estimators = n_estimators
        self.n_estimators_redundancy = n_estimators_redundancy

    def get_alphabet_size(self) -> int:
        """Get the alphabet size to use for the codebook.

        Returns:
            int: The alphabet size to use
        """
        if self.alphabet_size is None:
            return self.estimator.max_num_classes_
        return self.alphabet_size

    def get_n_estimators(self, n_classes: int) -> int:
        """Calculate the number of estimators to use based on the number of classes.

        Args:
            n_classes: The number of classes in the classification problem

        Returns:
            int: The number of estimators to use
        """
        if self.n_estimators is None:
            return (
                math.ceil(1 + math.log(n_classes, self.get_alphabet_size()))
                * 2
                * self.n_estimators_redundancy
            )
        return self.n_estimators

    def _generate_codebook(self, n_classes, n_estimators, alphabet_size):
        """Generate an efficient codebook using the provided alphabet size.

        This function generates a codebook where for each codeword at most `alphabet_size - 1` classes
        are mapped to unique classes, and the remaining classes are mapped to the "rest" class. The codebook
        is generated to provide for each class a unique codeword with maximum reduncancy between codewords.
        Greedy optimization is used to find the optimal codewords.

        Parameters:
            n_classes : int
                Number of classes.
            n_estimators : int
                Number of estimators.
            alphabet_size : int, default=10
                Size of the alphabet for the codebook. The first `alphabet_size - 1` classes
                will be mapped to unique classes, and the remaining classes will be mapped
                to the "rest" class.

        Returns:
            codebook : ndarray of shape (n_estimators, n_classes)
                Efficient n-ary codebook.
        """
        n_classes_in_codebook = min(n_classes, alphabet_size - 1)
        n_classes_remaining = n_classes - n_classes_in_codebook

        # Initialize the codebook with zeros
        codebook = np.zeros((n_estimators, n_classes), dtype=int)

        def generate_codeword():
            choices = list(range(n_classes_in_codebook)) + [
                n_classes_in_codebook for _ in range(n_classes_remaining)
            ]
            return np.random.permutation(
                choices,
            )

        # Generate the first codeword randomly
        codebook[0] = generate_codeword()

        # Generate the remaining codewords
        for i in range(1, n_estimators):
            # Initialize the current codeword with random values
            current_codeword = generate_codeword()
            max_distance, n_min_distance_min, max_distance_codeword = (
                0,
                1000,
                current_codeword,
            )

            # Iteratively improve the current codeword
            for _ in range(1000):  # Number of iterations can be adjusted
                # Compute the Hamming distances between the current codeword and all previous codewords
                distances = np.sum(
                    (codebook[:i] == n_classes_in_codebook)
                    != (current_codeword == n_classes_in_codebook),
                    axis=0,
                )

                # Find the minimum Hamming distance
                min_distance = np.min(distances, axis=0)

                n_min_distance = np.sum(distances == min_distance)
                # print(min_distance, n_min_distance)  # noqa: ERA001

                if min_distance > max_distance or (
                    min_distance == max_distance and n_min_distance < n_min_distance_min
                ):
                    max_distance_codeword = current_codeword
                    max_distance = min_distance
                    n_min_distance_min = n_min_distance

                current_codeword = generate_codeword()

            # Assign the optimized codeword to the codebook
            codebook[i] = max_distance_codeword

        return codebook

    def fit(self, X, y, **fit_params):
        """Fit underlying estimators.

        Parameters:
            X : {array-like, sparse matrix} of shape (n_samples, n_features)
                Data.

            y : array-like of shape (n_samples,)
                Multi-class targets.

            **fit_params : dict
                Parameters passed to the ``estimator.fit`` method of each
                sub-estimator.

        Returns:
            self : object
                Returns a fitted instance of self.
        """
        y = self._validate_data(X="no_validation", y=y)

        check_random_state(self.random_state)
        check_classification_targets(y)

        self.classes_ = np.unique(y)
        n_classes = self.classes_.shape[0]
        if n_classes == 0:
            raise ValueError(
                "DeciaryOutputCodeClassifier can not be fit when no class is present.",
            )

        self.no_mapping_needed_ = False
        if n_classes <= self.get_alphabet_size():
            self.no_mapping_needed_ = True
            return self.estimator.fit(X, y, **fit_params)

        # Generate efficient deciary codebook
        self.code_book_ = self._generate_codebook(
            n_classes,
            self.get_n_estimators(n_classes),
            alphabet_size=self.get_alphabet_size(),
        )

        self.classes_index_ = {c: i for i, c in enumerate(self.classes_)}

        self.Y_train = np.array(
            [self.code_book_[:, y[i]] for i in range(_num_samples(y))],
            dtype=int,
        )
        self.X_train = X
        # print('CLF DIST IN', list(zip(np.unique(y), (np.unique(y, return_counts=True)[1] / y.shape[0]).tolist())))  # noqa: ERA001

        return self

    def predict_proba(self, X):
        """Predict probabilities using the underlying estimators.

        Parameters:
            X : {array-like, sparse matrix} of shape (n_samples, n_features)
                Data.

        Returns:
            p : ndarray of shape (n_samples, n_classes)
                Returns the probability of the samples for each class in the model,
                where classes are ordered as they are in `self.classes_`.
        """
        check_is_fitted(self)

        if self.no_mapping_needed_:
            return self.estimator.predict_proba(X)

        Y = np.array(
            [
                _fit_and_predict_proba(
                    self.estimator,
                    self.X_train,
                    self.Y_train[:, i],
                    X,
                )
                for i in range(self.code_book_.shape[0])
            ],
            dtype=np.float64,
        )

        # Y shape is (n_estimators, n_samples, n_classes_per_code)
        n_estimators, n_samples, n_classes_per_code = Y.shape
        rest_class = np.max(self.code_book_)

        # Compute the weighted probabilities for each class
        probabilities = np.zeros((n_samples, self.classes_.shape[0]))

        for i in range(n_estimators):
            for j in range(self.classes_.shape[0]):
                if self.code_book_[i, j] != rest_class:
                    j_remapped = self.code_book_[i, j]
                    probabilities[:, j] += Y[
                        i,
                        :,
                        j_remapped,
                    ]  # / (1 - Y[i, :, rest_class])
                    # print(j, Y[i, :, j_remapped])  # noqa: ERA001

        assert not ((self.code_book_ != rest_class).sum(0) == 0).any(), (
            f"Some classes are not mapped to any codeword. {self.code_book_} {self.classes_} {((self.code_book_ != rest_class).sum(0) == 0)}"
        )

        # Normalize the weighted probabilities to get the final class probabilities
        probabilities /= (self.code_book_ != rest_class).sum(0)
        probabilities /= probabilities.sum(axis=1, keepdims=True)

        return probabilities

    def predict(self, X):
        """Predict multi-class targets using underlying estimators.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Data.

        Returns:
        -------
        y : ndarray of shape (n_samples,)
            Predicted multi-class targets.
        """
        check_is_fitted(self)
        X = check_array(X)

        if self.no_mapping_needed_:
            return self.estimator.predict(X)

        # Get predicted probabilities from each model
        probas = self.predict_proba(X)

        # Return class with highest probability
        return self.classes_[np.argmax(probas, axis=1)]

    def set_categorical_features(self, categorical_features):
        """Set categorical features for the base estimator if it supports it.

        Parameters
        ----------
        categorical_features : list
            List of categorical feature indices
        """
        self.categorical_features = categorical_features
        if hasattr(self.estimator, "set_categorical_features"):
            self.estimator.set_categorical_features(categorical_features)


def _fit_and_predict_proba(
    estimator: BaseEstimator,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X: np.ndarray,
) -> np.ndarray:
    """Fit a cloned base estimator and predict probabilities for a single sub-problem.

    This helper function handles training and prediction for a single base
    estimator in the ManyClassClassifier ensemble. It clones the original
    estimator to prevent modification, fits it on the training data with the
    encoded labels for this specific sub-problem, and returns probability
    predictions for the test data.

    Parameters
    ----------
    estimator : estimator object
        Base estimator to clone and fit. Usually a TabPFNClassifier instance.

    X_train : array-like of shape (n_samples, n_features)
        Training data features used to fit the cloned estimator.

    Y_train : array-like of shape (n_samples,)
        Encoded target values for this particular sub-problem.
        These are derived from the codebook mapping.

    X : array-like of shape (n_samples_test, n_features)
        Test data for which to predict probabilities.

    Returns:
    -------
    probas : ndarray of shape (n_samples_test, n_classes)
        Predicted probabilities for each class in this sub-problem.

    Raises:
    ------
    AttributeError
        If the base estimator doesn't implement predict_proba.
    """
    # Clone the estimator to avoid modifying the original
    cloned_estimator = clone(estimator)

    # Fit the cloned estimator on this particular encoding of the target
    cloned_estimator.fit(X_train, Y_train)

    # Generate probability predictions
    if hasattr(cloned_estimator, "predict_proba"):
        return cloned_estimator.predict_proba(X)
    raise AttributeError("Base estimator must implement the predict_proba method.")
