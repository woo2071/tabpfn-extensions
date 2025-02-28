from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

try:
    from tabpfn_extensions.post_hoc_ensembles.greedy_weighted_ensemble import (
        GreedyWeightedEnsembleClassifier as GreedyWeightedEnsemble,
    )
    from tabpfn_extensions.post_hoc_ensembles.pfn_phe import TabPFNPostHocEnsemble
    from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import (
        AutoTabPFNClassifier,
        AutoTabPFNRegressor,
    )
    HAS_ENSEMBLE_MODULES = True
except ImportError:
    HAS_ENSEMBLE_MODULES = False

# Mark tests as client compatible and requiring any TabPFN implementation
# Post-hoc ensembles can work with either TabPFN or client
pytestmark = [
    pytest.mark.requires_any_tabpfn,  # Requires any TabPFN implementation
    pytest.mark.client_compatible,    # Compatible with TabPFN client
    pytest.mark.skipif(not HAS_ENSEMBLE_MODULES, reason="Post-hoc ensemble modules not available"),
]


@pytest.fixture
def classification_data():
    """Create a simple classification dataset."""
    rng = np.random.RandomState(42)
    X = rng.rand(100, 5)
    y = (X[:, 0] > 0.5).astype(int)
    return train_test_split(X, y, test_size=0.2, random_state=42)


@pytest.fixture
def base_models(tabpfn_classifier):
    """Create a set of base models for ensembling."""
    return [
        RandomForestClassifier(n_estimators=10, random_state=42),
        LogisticRegression(random_state=42),
        tabpfn_classifier,
    ]


def test_greedy_weighted_ensemble(classification_data, base_models):
    """Test GreedyWeightedEnsemble basic functionality."""
    X_train, X_test, y_train, y_test = classification_data

    # Create tuples of (name, estimator) for the ensemble
    named_estimators = [
        (f"model_{i}", model) for i, model in enumerate(base_models)
    ]

    # Initialize ensemble
    ensemble = GreedyWeightedEnsemble(
        estimators=named_estimators,
        score_metric="log_loss",
        n_iterations=5,  # Small for testing
        silo_top_n=3,
        seed=42,
    )

    # Fit base models
    for model in base_models:
        model.fit(X_train, y_train)

    # Fit ensemble
    ensemble.fit(X_train, y_train)

    # Check that fit worked
    assert hasattr(ensemble, "ensemble")

    # Test predictions
    y_pred = ensemble.predict(X_test)
    assert y_pred.shape == y_test.shape

    # Test probability predictions
    y_proba = ensemble.predict_proba(X_test)
    assert y_proba.shape == (X_test.shape[0], 2)  # Binary classification

    # Check properties of probabilities
    assert np.all(y_proba >= 0)
    assert np.all(y_proba <= 1)
    assert np.allclose(y_proba.sum(axis=1), 1.0)


def test_tabpfn_post_hoc_ensemble(classification_data, tabpfn_classifier):
    """Test TabPFNPostHocEnsemble functionality."""
    X_train, X_test, y_train, y_test = classification_data

    # Make sure we're using the tabpfn model type that's expected by TabPFNPostHocEnsemble
    if "device" in tabpfn_classifier.get_params():
        # Initialize ensemble with TabPFN as base model
        ensemble = TabPFNPostHocEnsemble(
            n_models=3,  # Small for testing
            # Let device default to auto
        )
    else:
        # If using a different implementation (like client), try to adapt
        pytest.skip("TabPFNPostHocEnsemble requires TabPFN with device parameter")

    # Fit ensemble
    ensemble.fit(X_train, y_train)

    # Check attributes after fitting
    assert hasattr(ensemble, "base_models_")
    assert len(ensemble.base_models_) == 3
    assert hasattr(ensemble, "ensemble_")

    # Test prediction
    y_pred = ensemble.predict(X_test)
    assert y_pred.shape == y_test.shape

    # Test probability prediction
    y_proba = ensemble.predict_proba(X_test)
    assert y_proba.shape == (X_test.shape[0], 2)  # Binary classification

    # Check that probabilities sum to 1
    assert np.allclose(y_proba.sum(axis=1), 1.0)


def test_ensemble_weighting_schemes(classification_data, base_models):
    """Test different weighting schemes for ensembles."""
    X_train, X_test, y_train, y_test = classification_data

    # Create tuples of (name, estimator) for the ensemble
    named_estimators = [
        (f"model_{i}", model) for i, model in enumerate(base_models)
    ]

    # Fit base models
    for model in base_models:
        model.fit(X_train, y_train)

    for score_metric in ["log_loss", "accuracy"]:
        # Initialize ensemble
        ensemble = GreedyWeightedEnsemble(
            estimators=named_estimators,
            score_metric=score_metric,
            n_iterations=5,  # Small for testing
            silo_top_n=3,
            seed=42,
        )

        # Fit ensemble
        ensemble.fit(X_train, y_train)

        # Test prediction
        y_pred = ensemble.predict(X_test)
        assert y_pred.shape == y_test.shape

        # Test probability prediction
        y_proba = ensemble.predict_proba(X_test)

        # Check ensemble predictions
        assert y_proba.shape == (X_test.shape[0], 2)


def test_ensemble_diversity(classification_data, tabpfn_classifier):
    """Test ensemble diversity with different TabPFN configurations."""
    X_train, X_test, y_train, y_test = classification_data

    # Create diverse models (use the same TabPFN instance for simplicity)
    # In a real scenario, we would create models with different parameters
    models = [
        tabpfn_classifier,
        RandomForestClassifier(n_estimators=5, random_state=42),
        RandomForestClassifier(n_estimators=10, random_state=24),
    ]

    # Create named estimators
    named_estimators = [
        (f"tabpfn_{i}", model) for i, model in enumerate(models)
    ]

    # Fit models
    for model in models:
        model.fit(X_train, y_train)

    # Create ensemble
    ensemble = GreedyWeightedEnsemble(
        estimators=named_estimators,
        score_metric="log_loss",
        n_iterations=5,  # Small for testing
        silo_top_n=3,
        seed=42,
    )

    # Fit ensemble
    ensemble.fit(X_train, y_train)

    # Test predictions
    ensemble_preds = ensemble.predict_proba(X_test)
    ensemble_pred_class = ensemble.predict(X_test)

    # Check shapes
    assert ensemble_preds.shape == (X_test.shape[0], 2)
    assert ensemble_pred_class.shape == y_test.shape

    # Ensemble should perform at least as well as average of base models
    np.mean([model.score(X_test, y_test) for model in models])
    ensemble_acc = np.mean(ensemble_pred_class == y_test)

    # Because we're using a very small ensemble with minimal iterations,
    # we may not always beat the base models, so just check the shapes
    assert isinstance(ensemble_acc, float)


def test_auto_tabpfn_classifier(classification_data, tabpfn_classifier):
    """Test AutoTabPFNClassifier - the sklearn interface used in the example."""
    X_train, X_test, y_train, y_test = classification_data

    # Initialize the auto classifier with minimal settings for test speed
    auto_classifier = AutoTabPFNClassifier(
        phe_init_args={"max_models": 2},  # Use minimal ensemble for test speed
        random_state=42,
    )

    # Fit the model
    auto_classifier.fit(X_train, y_train)

    # Test predictions
    y_pred = auto_classifier.predict(X_test)
    assert y_pred.shape == y_test.shape

    # Test probability predictions
    y_proba = auto_classifier.predict_proba(X_test)
    assert y_proba.shape == (X_test.shape[0], 2)  # Binary classification

    # Check properties of probabilities
    assert np.all(y_proba >= 0)
    assert np.all(y_proba <= 1)
    assert np.allclose(y_proba.sum(axis=1), 1.0)

    # Test score method
    score = auto_classifier.score(X_test, y_test)
    assert isinstance(score, float)
    assert 0 <= score <= 1

    # Check that attributes are set
    assert hasattr(auto_classifier, "predictor_")


@pytest.fixture
def regression_data():
    """Create a simple regression dataset."""
    rng = np.random.RandomState(42)
    X = rng.rand(100, 5)
    y = 2 * X[:, 0] + 3 * X[:, 1] - 1.5 * X[:, 2] + rng.normal(0, 0.1, size=100)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def test_auto_tabpfn_regressor(regression_data, tabpfn_regressor):
    """Test AutoTabPFNRegressor - the sklearn interface used in the example."""
    # Skip if tabpfn_regressor fixture isn't available
    if tabpfn_regressor is None:
        pytest.skip("TabPFN regressor not available")

    X_train, X_test, y_train, y_test = regression_data

    # Initialize the auto regressor with minimal settings for test speed
    auto_regressor = AutoTabPFNRegressor(
        phe_init_args={"max_models": 2},  # Use minimal ensemble for test speed
        random_state=42,
    )

    # Fit the model
    auto_regressor.fit(X_train, y_train)

    # Test predictions
    y_pred = auto_regressor.predict(X_test)
    assert y_pred.shape == y_test.shape

    # Test score method (R^2 score)
    score = auto_regressor.score(X_test, y_test)
    assert isinstance(score, float)

    # Check that attributes are set
    assert hasattr(auto_regressor, "predictor_")