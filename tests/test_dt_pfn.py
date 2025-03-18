"""Tests for the TabPFN-based Decision Tree implementation.

This file tests the Decision Tree TabPFN implementations in tabpfn_extensions.rf_pfn.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, r2_score

from conftest import FAST_TEST_MODE
from tabpfn_extensions.rf_pfn.sklearn_based_decision_tree_tabpfn import (
    DecisionTreeTabPFNClassifier,
    DecisionTreeTabPFNRegressor,
)
from test_base_tabpfn import BaseClassifierTests, BaseRegressorTests


@pytest.mark.requires_tabpfn
class TestDecisionTreeClassifier(BaseClassifierTests):
    """Test DecisionTreeTabPFNClassifier using the BaseClassifierTests framework."""

    @pytest.fixture
    def estimator(self, tabpfn_classifier):
        """Provide a TabPFN-based DecisionTreeClassifier as the estimator."""
        return DecisionTreeTabPFNClassifier(
            tabpfn=tabpfn_classifier,
            max_depth=3,  # Shallow tree for speed
            random_state=42,
        )

    # Skip pandas and text feature tests as they're not fully supported by DecisionTreeTabPFN
    @pytest.mark.skip(
        reason="DecisionTreeTabPFN doesn't fully support pandas DataFrames",
    )
    def test_with_pandas(self, estimator, pandas_classification_data):
        pass

    @pytest.mark.skip(reason="DecisionTreeTabPFN doesn't fully support text features")
    def test_with_text_features(self, estimator, dataset_generator):
        pass

    def test_special_case_behavior(self, estimator, classification_data):
        """Test special behavior in the decision tree implementation."""
        X, y = classification_data

        # Original TabPFN fit and predict
        original_tabpfn = estimator.tabpfn
        original_tabpfn.fit(X, y)
        tabpfn_preds = original_tabpfn.predict(X)

        # Very shallow tree (min_samples_split set high) should give similar results
        # as it will just use TabPFN at the root
        estimator.min_samples_split = X.shape[0]  # Force a single node
        estimator.fit(X, y)
        y_pred = estimator.predict(X)

        # Check predictions
        assert y_pred.shape == y.shape
        accuracy = accuracy_score(y, y_pred)
        assert accuracy > 0.6

        # Should be the same predictions as a direct TabPFN model
        assert np.array_equal(y_pred, tabpfn_preds)


@pytest.mark.requires_tabpfn
class TestDecisionTreeRegressor(BaseRegressorTests):
    """Test DecisionTreeTabPFNRegressor using the BaseRegressorTests framework."""

    @pytest.fixture
    def estimator(self, tabpfn_regressor):
        """Provide a TabPFN-based DecisionTreeRegressor as the estimator."""
        return DecisionTreeTabPFNRegressor(
            tabpfn=tabpfn_regressor,
            max_depth=3,  # Shallow tree for speed
            random_state=42,
        )

    # Skip pandas and text feature tests as they're not fully supported by DecisionTreeTabPFN
    @pytest.mark.skip(
        reason="DecisionTreeTabPFN doesn't fully support pandas DataFrames",
    )
    def test_with_pandas(self, estimator, pandas_regression_data):
        pass

    @pytest.mark.skip(reason="DecisionTreeTabPFN doesn't fully support text features")
    def test_with_text_features(self, estimator, dataset_generator):
        pass

    def test_tree_model_behavior(self, estimator, regression_data):
        """Test basic tree behavior."""
        X, y = regression_data

        # Fit with default parameters
        estimator.fit(X, y)

        # Make predictions
        y_pred = estimator.predict(X)

        # Check predictions
        assert y_pred.shape == y.shape
        r2 = r2_score(y, y_pred)
        assert r2 > 0.5

        # Try with a different max_depth
        if estimator.max_depth is not None:
            original_max_depth = estimator.max_depth
            estimator.max_depth = 2
        else:
            original_max_depth = None
            estimator.max_depth = 2

        # Should be able to refit
        estimator.fit(X, y)
        y_pred_shallow = estimator.predict(X)

        # Results should be different with a different tree depth
        assert not np.array_equal(y_pred, y_pred_shallow)

        # Restore original max_depth
        estimator.max_depth = original_max_depth


@pytest.mark.requires_tabpfn
class TestDecisionTreeAdvanced:
    """Advanced tests for TabPFN Decision Trees that go beyond the base test cases."""

    def test_adaptive_tree_classifier(self, tabpfn_classifier, classification_data):
        """Test adaptive tree functionality for classifiers."""
        if FAST_TEST_MODE:
            pytest.skip("Skipping adaptive tree test in fast mode")

        X, y = classification_data

        # Create a tree with adaptive node fitting enabled
        dt_clf = DecisionTreeTabPFNClassifier(
            tabpfn=tabpfn_classifier,
            max_depth=2,
            adaptive_tree=True,
            adaptive_tree_min_train_samples=10,
            adaptive_tree_test_size=0.2,
            random_state=42,
        )

        dt_clf.fit(X, y)
        y_pred = dt_clf.predict(X)

        # Check that predictions are reasonable
        accuracy = accuracy_score(y, y_pred)
        assert accuracy > 0.6

    def test_adaptive_tree_regressor(self, tabpfn_regressor, regression_data):
        """Test adaptive tree functionality for regressors."""
        if FAST_TEST_MODE:
            pytest.skip("Skipping adaptive tree test in fast mode")

        X, y = regression_data

        # Create a tree with adaptive node fitting enabled
        dt_reg = DecisionTreeTabPFNRegressor(
            tabpfn=tabpfn_regressor,
            max_depth=2,
            adaptive_tree=True,
            adaptive_tree_min_train_samples=10,
            adaptive_tree_test_size=0.2,
            random_state=42,
        )

        dt_reg.fit(X, y)
        y_pred = dt_reg.predict(X)

        # Check that predictions are reasonable
        r2 = r2_score(y, y_pred)
        assert r2 > 0.5

    def test_different_max_depths_classifier(
        self,
        tabpfn_classifier,
        classification_data,
    ):
        """Test classifier with different max_depth values."""
        if FAST_TEST_MODE:
            pytest.skip("Skipping max_depth test in fast mode")

        X, y = classification_data

        # Test different tree depths
        depths = [1, 2, 3] if FAST_TEST_MODE else [1, 2, 3, 5]
        accuracies = []

        for depth in depths:
            dt_clf = DecisionTreeTabPFNClassifier(
                tabpfn=tabpfn_classifier,
                max_depth=depth,
                random_state=42,
            )

            dt_clf.fit(X, y)
            y_pred = dt_clf.predict(X)

            # Track accuracy
            accuracies.append(accuracy_score(y, y_pred))

        # Generally, deeper trees should have higher accuracy on the training data
        for i in range(1, len(accuracies)):
            assert (
                accuracies[i] >= accuracies[i - 1] - 0.05
            )  # Allow small decrease due to randomness
