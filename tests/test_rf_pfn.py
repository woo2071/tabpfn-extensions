from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

try:
    from tabpfn_extensions.rf_pfn import (
        DecisionTreeTabPFNClassifier,
        DecisionTreeTabPFNRegressor,
        RandomForestTabPFNClassifier,
        RandomForestTabPFNRegressor,
    )

    HAS_RF_PFN = True
except ImportError:
    HAS_RF_PFN = False

try:
    from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor

    HAS_TABPFN = True
except ImportError:
    try:
        from tabpfn import TabPFNClassifier, TabPFNRegressor

        HAS_TABPFN = True
    except ImportError:
        try:
            from tabpfn_client import TabPFNClassifier, TabPFNRegressor

            HAS_TABPFN = True
        except ImportError:
            HAS_TABPFN = False


# Import common testing utilities from conftest
from conftest import DEFAULT_TEST_SIZE, FAST_TEST_MODE, SMALL_TEST_SIZE

# Skip all tests if RF_PFN is not available
pytestmark = [
    pytest.mark.skipif(
        not HAS_RF_PFN or not HAS_TABPFN,
        reason="RF_PFN module or TabPFN not available",
    ),
    pytest.mark.requires_any_tabpfn,  # Requires any TabPFN implementation
    pytest.mark.client_compatible,  # Compatible with TabPFN client
]

# Number of samples to use in tests
N_SAMPLES = SMALL_TEST_SIZE if FAST_TEST_MODE else DEFAULT_TEST_SIZE


@pytest.mark.parametrize(
    "model_class",
    [
        (RandomForestTabPFNClassifier, TabPFNClassifier),
        (RandomForestTabPFNRegressor, TabPFNRegressor),
        (DecisionTreeTabPFNClassifier, TabPFNClassifier),
        (DecisionTreeTabPFNRegressor, TabPFNRegressor),
    ],
)
def test_sklearn_compatibility(
    model_class: tuple[type[BaseEstimator], type[TabPFNClassifier | TabPFNRegressor]],
) -> None:
    """Test RandomForestTabPFN compatibility with different sklearn versions.

    Args:
        model_class: Tuple of (RF/DT model class, base TabPFN class)
    """
    # Generate sample data - smaller in fast mode
    rng = np.random.RandomState(42)
    X = rng.rand(N_SAMPLES, 4)
    if model_class[1] == TabPFNClassifier:
        y = rng.randint(0, 2, N_SAMPLES)
    else:
        y = rng.randn(N_SAMPLES)

    # Initialize classifier - minimal model settings for fast testing
    clf_class, clf_base_class = model_class
    if "RandomForest" in clf_class.__name__:
        # Use very small forest in fast mode
        n_trees = 1 if FAST_TEST_MODE else 2
        kwargs = {
            "tabpfn": clf_base_class(),  # Let device default to auto
            "n_estimators": n_trees,
            "max_depth": 2,
        }
    else:
        # Decision Tree settings
        kwargs = {
            "tabpfn": clf_base_class(),  # Let device default to auto
            "max_depth": 2,
        }

    clf = clf_class(**kwargs)

    # This should work without errors on both sklearn <1.6 and >=1.6
    clf.fit(X, y)

    # Verify predictions work
    pred = clf.predict(X)
    assert pred.shape == (N_SAMPLES,)
    if clf_base_class == TabPFNClassifier:
        assert np.all(np.isin(pred, [0, 1]))
    else:
        assert pred.dtype == np.float64


@pytest.mark.parametrize(
    "model_class",
    [
        (RandomForestTabPFNClassifier, TabPFNClassifier),
        (RandomForestTabPFNRegressor, TabPFNRegressor),
        (DecisionTreeTabPFNClassifier, TabPFNClassifier),
        (DecisionTreeTabPFNRegressor, TabPFNRegressor),
    ],
)
def test_with_nan(model_class):
    """Test that models can handle NaN values in data."""
    # Generate sample data with NaN values - smaller in fast mode
    rng = np.random.RandomState(42)
    X = rng.rand(N_SAMPLES, 4)
    X[0, 0] = np.nan  # Add a NaN value

    # Create appropriate target type
    if model_class[1] == TabPFNClassifier:
        y = rng.randint(0, 2, N_SAMPLES)
    else:
        y = rng.randn(N_SAMPLES)

    # Initialize model with minimal settings for speed
    clf_class, clf_base_class = model_class
    if "RandomForest" in clf_class.__name__:
        # Use smaller forest in fast mode
        n_trees = 1 if FAST_TEST_MODE else 2
        kwargs = {
            "tabpfn": clf_base_class(),  # Let device default to auto
            "n_estimators": n_trees,
            "max_depth": 2,
        }
    else:
        # Decision Tree settings
        kwargs = {
            "tabpfn": clf_base_class(),  # Let device default to auto
            "max_depth": 2,
        }

    clf = clf_class(**kwargs)

    # This should work without errors
    clf.fit(X, y)

    # Test prediction with NaN values
    X_test = X.copy()
    pred = clf.predict(X_test)

    assert pred.shape == (N_SAMPLES,)
    if clf_base_class == TabPFNClassifier:
        assert np.all(np.isin(pred, [0, 1]))
    else:
        assert pred.dtype == np.float64


def test_binary_classification():
    """Test RandomForestTabPFNClassifier with binary data."""
    # Skip if we don't have the right modules
    if not HAS_RF_PFN or not HAS_TABPFN:
        pytest.skip("RF_PFN module or TabPFN not available")

    # Create a synthetic binary classification problem
    X, y = make_classification(
        n_samples=30,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        n_classes=2,
        random_state=42,
    )

    # Split data ensuring both classes are represented
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.33,
        random_state=42,
        stratify=y,
    )

    # Verify we have both classes
    assert len(np.unique(y_train)) == 2
    assert len(np.unique(y_test)) == 2

    # Create base TabPFN classifier
    base_clf = TabPFNClassifier()  # Let device default to auto

    # Create RF-TabPFN classifier with minimal settings for test speed
    rf_clf = RandomForestTabPFNClassifier(
        tabpfn=base_clf,
        n_estimators=2,  # Minimal for testing
        max_depth=3,
        max_predict_time=5,  # Short timeout
    )

    # Train model
    rf_clf.fit(X_train, y_train)

    # Test predictions
    y_pred = rf_clf.predict(X_test)
    assert y_pred.shape == y_test.shape

    # Test probabilities
    y_proba = rf_clf.predict_proba(X_test)
    assert y_proba.shape == (len(X_test), 2)  # 2 classes
    assert np.allclose(y_proba.sum(axis=1), 1.0)

    # Verify score works
    score = rf_clf.score(X_test, y_test)
    assert isinstance(score, float)
    assert 0 <= score <= 1


def test_regression_performance():
    """Test RandomForestTabPFNRegressor prediction performance, similar to the example."""
    # Skip if we don't have the right modules
    if not HAS_RF_PFN or not HAS_TABPFN:
        pytest.skip("RF_PFN module or TabPFN not available")

    # Create a synthetic regression dataset
    X, y = make_regression(
        n_samples=50,
        n_features=5,
        n_informative=3,
        random_state=42,
    )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.33,
        random_state=42,
    )

    # Create base TabPFN regressor
    base_reg = TabPFNRegressor()  # Let device default to auto

    # Create RF-TabPFN regressor with minimal settings for test speed
    rf_reg = RandomForestTabPFNRegressor(
        tabpfn=base_reg,
        n_estimators=2,  # Minimal for testing
        max_depth=3,
        max_predict_time=5,  # Short timeout
    )

    # Train model
    rf_reg.fit(X_train, y_train)

    # Test predictions
    y_pred = rf_reg.predict(X_test)
    assert y_pred.shape == y_test.shape

    # Check MSE and R^2
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Just check types, not specific values
    assert isinstance(mse, float)
    assert isinstance(r2, float)
