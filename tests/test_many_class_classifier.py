import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split

from tabpfn_extensions.many_class import ManyClassClassifier


@pytest.fixture
def sample_binary_data():
    """Generate sample binary classification data."""
    X, y = make_classification(
        n_samples=100, n_features=10, n_classes=2, random_state=42
    )
    return train_test_split(X, y, test_size=0.2, random_state=42)


@pytest.fixture
def sample_multiclass_data():
    """Generate sample multiclass classification data."""
    X, y = make_classification(
        n_samples=200, n_features=15, n_classes=5, n_informative=10, random_state=42
    )
    return train_test_split(X, y, test_size=0.2, random_state=42)


def test_initialization():
    """Test initialization with various parameters."""
    base_estimator = RandomForestClassifier()

    # Test default initialization
    clf = ManyClassClassifier(estimator=base_estimator)
    assert clf.n_estimators is None
    assert clf.alphabet_size is None

    # Test custom parameters
    clf = ManyClassClassifier(
        estimator=base_estimator, alphabet_size=5, n_estimators=10, random_state=42
    )
    assert clf.alphabet_size == 5
    assert clf.n_estimators == 10
    assert clf.random_state == 42


def test_binary_classification(sample_binary_data):
    """Test with binary classification problem."""
    X_train, X_test, y_train, y_test = sample_binary_data
    base_estimator = RandomForestClassifier(random_state=42)
    clf = ManyClassClassifier(estimator=base_estimator, random_state=42)

    # Fit and predict
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Check predictions shape and values
    assert y_pred.shape == y_test.shape
    assert np.all(np.unique(y_pred) == np.unique(y_test))

    # Check probabilities
    proba = clf.predict_proba(X_test)
    assert proba.shape == (len(y_test), 2)
    assert np.allclose(np.sum(proba, axis=1), 1.0)


def test_multiclass_classification(sample_multiclass_data):
    """Test with multiclass classification problem."""
    X_train, X_test, y_train, y_test = sample_multiclass_data
    base_estimator = RandomForestClassifier(random_state=42)

    # Test with different alphabet sizes
    for alphabet_size in [3, 5, 10]:
        clf = ManyClassClassifier(
            estimator=base_estimator, alphabet_size=alphabet_size, random_state=42
        )

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Basic checks
        assert y_pred.shape == y_test.shape
        assert len(np.unique(y_pred)) == len(np.unique(y_test))

        # Probability checks
        proba = clf.predict_proba(X_test)
        assert proba.shape == (len(y_test), len(np.unique(y_test)))
        assert np.allclose(np.sum(proba, axis=1), 1.0)


def test_error_handling():
    """Test error handling and edge cases."""
    clf = ManyClassClassifier(estimator=RandomForestClassifier())

    # Test prediction before fitting
    X = np.random.rand(10, 5)
    with pytest.raises(NotFittedError):
        clf.predict(X)

    # Test empty data
    with pytest.raises(ValueError):
        clf.fit(np.array([]), np.array([]))

    # Test single class
    X = np.random.rand(10, 5)
    y = np.zeros(10)
    with pytest.raises(ValueError):
        clf.fit(X, y)


def test_codebook_generation():
    """Test codebook generation functionality."""
    base_estimator = RandomForestClassifier(random_state=42)
    clf = ManyClassClassifier(
        estimator=base_estimator, alphabet_size=5, random_state=42
    )

    # Generate synthetic data with many classes
    X, y = make_classification(
        n_samples=200, n_features=15, n_classes=10, random_state=42
    )

    clf.fit(X, y)

    # Check codebook properties
    assert hasattr(clf, "code_book_")
    assert clf.code_book_.shape[1] == len(np.unique(y))
    assert set(np.unique(clf.code_book_)) <= set(range(clf.alphabet_size))


def test_categorical_features():
    """Test handling of categorical features."""
    base_estimator = RandomForestClassifier(random_state=42)
    clf = ManyClassClassifier(estimator=base_estimator)

    # Create data with categorical features
    X = np.random.rand(100, 5)
    X[:, [1, 3]] = np.random.randint(0, 3, size=(100, 2))
    y = np.random.randint(0, 4, size=100)

    # Set categorical features
    categorical_features = [1, 3]
    clf.set_categorical_features(categorical_features)

    # Verify the categorical features are set
    assert hasattr(clf, "categorical_features")
    assert clf.categorical_features == categorical_features


if __name__ == "__main__":
    pytest.main([__file__])
