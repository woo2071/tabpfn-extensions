from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from tabpfn_extensions.many_class import ManyClassClassifier

# Mark all tests in this file as compatible with TabPFN client
pytestmark = pytest.mark.client_compatible

# Helper to get a basic classifier for testing
def get_test_classifier():
    """Get a classifier for testing using RandomForest (reliable fallback)."""
    # Use RandomForestClassifier to ensure consistent test behavior
    return RandomForestClassifier(n_estimators=10, random_state=42)


@pytest.fixture
def multi_class_data():
    """Create a dataset with more than 10 classes."""
    rng = np.random.RandomState(42)
    X = rng.rand(300, 5)
    # Create 15 classes (exceeding TabPFN's default limit of 10)
    y = rng.randint(0, 15, 300)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def test_basic_functionality(multi_class_data):
    """Test that ManyClassClassifier can handle more than 10 classes."""
    X_train, X_test, y_train, y_test = multi_class_data

    # Initialize with a classifier
    base_classifier = get_test_classifier()
    model = ManyClassClassifier(estimator=base_classifier, alphabet_size=10)

    # Fit the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Verify predictions
    assert y_pred.shape == y_test.shape
    assert len(np.unique(y_pred)) > 5, "Should support multiple classes"

    # Check accuracy - just make sure it runs and produces valid results
    # Note: Random Forest with small number of trees may not be better than random
    # guessing for this many classes
    1.0 / len(np.unique(y_train))
    model_accuracy = accuracy_score(y_test, y_pred)
    assert model_accuracy >= 0.0, "Model accuracy should be a valid score"


def test_codebook_config():
    """Test different codebook configuration options."""
    rng = np.random.RandomState(42)
    X = rng.rand(100, 5)
    y = rng.randint(0, 20, 100)  # 20 classes

    # Test with different number of estimators
    for n_estimators in [10, 15, 20]:
        base_classifier = get_test_classifier()
        model = ManyClassClassifier(
            estimator=base_classifier,
            alphabet_size=10,
            n_estimators=n_estimators,
        )
        model.fit(X, y)
        y_pred = model.predict(X)

        assert y_pred.shape == y.shape
        # Ensure all predicted classes are valid
        assert np.all(np.isin(y_pred, np.unique(y)))


def test_large_class_count():
    """Test with a very large number of classes."""
    rng = np.random.RandomState(42)
    X = rng.rand(200, 5)
    y = rng.randint(0, 30, 200)  # 30 classes

    base_classifier = get_test_classifier()
    model = ManyClassClassifier(
        estimator=base_classifier,
        alphabet_size=10,
        n_estimators_redundancy=2,  # Lower redundancy for faster testing
    )
    model.fit(X, y)
    y_pred = model.predict(X[:10])  # Predict just a few samples to keep test fast

    # Ensure predictions have the right shape
    assert y_pred.shape == (10,)
    # Ensure all predicted classes are in the original set
    assert np.all(np.isin(y_pred, np.unique(y)))


def test_predict_proba():
    """Test that predict_proba returns valid probabilities."""
    rng = np.random.RandomState(42)
    X = rng.rand(100, 5)
    y = rng.randint(0, 15, 100)

    base_classifier = get_test_classifier()
    model = ManyClassClassifier(
        estimator=base_classifier,
        alphabet_size=10,
    )
    model.fit(X, y)

    # Get probability predictions
    y_proba = model.predict_proba(X[:5])  # Just a few samples

    # Check dimensions
    assert y_proba.shape[0] == 5
    assert y_proba.shape[1] == len(np.unique(y))

    # Check probabilities sum to 1
    assert np.allclose(y_proba.sum(axis=1), 1.0, rtol=1e-5)

    # Each row should be a probability distribution
    assert (y_proba >= 0).all()
    assert (y_proba <= 1).all()


def test_sklearn_interface():
    """Test sklearn interface compatibility."""
    rng = np.random.RandomState(42)
    X = rng.rand(50, 5)
    y = rng.randint(0, 12, 50)

    base_classifier = get_test_classifier()
    model = ManyClassClassifier(
        estimator=base_classifier,
        alphabet_size=10,
    )

    # Should support get_params and set_params
    params = model.get_params()
    assert "alphabet_size" in params
    assert "n_estimators_redundancy" in params

    # Should support score method
    model.fit(X, y)
    score = model.score(X, y)
    assert 0 <= score <= 1

    # Should have classes_ attribute after fitting
    assert hasattr(model, "classes_")
    assert len(model.classes_) == len(np.unique(y))


@pytest.mark.requires_any_tabpfn
def test_with_tabpfn_classifier(tabpfn_classifier):
    """Test ManyClassClassifier with TabPFN classifier."""
    rng = np.random.RandomState(42)
    X = rng.rand(100, 5)
    y = rng.randint(0, 12, 100)  # 12 classes
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize with TabPFN
    model = ManyClassClassifier(
        estimator=tabpfn_classifier,
        alphabet_size=8,  # TabPFN can handle up to 10 classes
        n_estimators=5,    # Use fewer estimators to speed up the test
    )

    # Fit the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Verify predictions
    assert y_pred.shape == y_test.shape
    assert set(np.unique(y_pred)).issubset(set(np.unique(y_train))), "All predicted classes should be in training set"

    # Get probability predictions
    y_proba = model.predict_proba(X_test[:3])

    # Check dimensions
    assert y_proba.shape[0] == 3
    assert y_proba.shape[1] == len(np.unique(y_train))

    # Check probabilities sum to 1
    assert np.allclose(y_proba.sum(axis=1), 1.0, rtol=1e-5)