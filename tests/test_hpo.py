"""Tests for the TabPFN hyperparameter optimization (HPO) implementation.

This file tests the HPO implementations in tabpfn_extensions.hpo.
"""

from __future__ import annotations

import pytest

from conftest import FAST_TEST_MODE

# Try to import the HPO module, but skip tests if hyperopt is not available
try:
    from tabpfn_extensions.hpo import TunedTabPFNClassifier, TunedTabPFNRegressor

    HYPEROPT_AVAILABLE = True
except ImportError:
    HYPEROPT_AVAILABLE = False
    pytest.skip(
        "hyperopt not installed. Install with 'pip install \"tabpfn-extensions[hpo]\"'",
        allow_module_level=True,
    )

from test_base_tabpfn import BaseClassifierTests, BaseRegressorTests


@pytest.mark.requires_tabpfn
@pytest.mark.client_compatible
class TestTunedTabPFNClassifier(BaseClassifierTests):
    """Test TunedTabPFNClassifier using the BaseClassifierTests framework."""

    @pytest.fixture
    def estimator(self, tabpfn_classifier):
        """Provide a HPO-based TabPFN classifier as the estimator."""
        n_trials = 3 if FAST_TEST_MODE else 10  # Very limited trials for fast testing

        return TunedTabPFNClassifier(
            n_trials=n_trials,
            metric="accuracy",
            random_state=42,
        )

    @pytest.mark.skip(reason="Tuned TabPFN models take too long for this test")
    def test_with_various_datasets(self, estimator, dataset_generator):
        """Skip test with various datasets as it takes too long for HPO."""
        pass


@pytest.mark.requires_tabpfn
@pytest.mark.client_compatible
class TestTunedTabPFNRegressor(BaseRegressorTests):
    """Test TunedTabPFNRegressor using the BaseRegressorTests framework."""

    @pytest.fixture
    def estimator(self, tabpfn_regressor):
        """Provide a HPO-based TabPFN regressor as the estimator."""
        n_trials = 3 if FAST_TEST_MODE else 10  # Very limited trials for fast testing

        return TunedTabPFNRegressor(
            n_trials=n_trials,
            metric="rmse",
            random_state=42,
        )

    @pytest.mark.skip(reason="Tuned TabPFN models take too long for this test")
    def test_with_various_datasets(self, estimator, dataset_generator):
        """Skip test with various datasets as it takes too long for HPO."""
        pass


# Additional HPO-specific tests
class TestHPOSpecificFeatures:
    """Test HPO-specific features that aren't covered by the base tests."""

    @pytest.mark.requires_tabpfn
    @pytest.mark.client_compatible
    def test_different_metrics(self):
        """Test different metrics for HPO."""
        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import train_test_split

        # Load a dataset
        X, y = load_breast_cancer(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
        )

        # Create models with different metrics
        metrics = (
            ["accuracy", "roc_auc"] if FAST_TEST_MODE else ["accuracy", "roc_auc", "f1"]
        )

        for metric in metrics:
            model = TunedTabPFNClassifier(
                n_trials=3 if FAST_TEST_MODE else 10,
                metric=metric,
                random_state=42,
            )

            # Fit and predict
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Just check that it runs without errors
            assert y_pred.shape == y_test.shape

            # Check that the best params and score are stored
            assert hasattr(model, "best_params_")
            assert hasattr(model, "best_score_")
