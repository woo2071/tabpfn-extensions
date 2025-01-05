import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from tabpfn_extensions.rf_pfn import (
    RandomForestTabPFNClassifier,
    RandomForestTabPFNRegressor,
)


@pytest.fixture
def binary_data():
    X, y = load_breast_cancer(return_X_y=True)
    return X[:100], y[:100]  # Using subset for faster testing


@pytest.fixture
def multiclass_data():
    X, y = load_iris(return_X_y=True)
    return X[:100], y[:100]


@pytest.fixture
def regression_data():
    X, y = load_diabetes(return_X_y=True)
    return X[:100], y[:100]


class TestRandomForestTabPFNClassifier:
    def test_init(self):
        clf = RandomForestTabPFNClassifier()
        assert clf.max_time == 60
        assert clf.n_estimators == 10
        assert clf.random_state is None

    def test_binary_classification(self, binary_data):
        X, y = binary_data
        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]

        clf = RandomForestTabPFNClassifier(random_state=42)
        clf.fit(X_train, y_train)

        # Test predict
        predictions = clf.predict(X_test)
        assert predictions.shape == (20,)
        assert np.all(np.isin(predictions, [0, 1]))

        # Test predict_proba
        proba = clf.predict_proba(X_test)
        assert proba.shape == (20, 2)
        assert np.allclose(np.sum(proba, axis=1), 1.0)
        assert np.all((proba >= 0) & (proba <= 1))

    def test_multiclass_classification(self, multiclass_data):
        X, y = multiclass_data
        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]

        clf = RandomForestTabPFNClassifier(random_state=42)
        clf.fit(X_train, y_train)

        # Test predict
        predictions = clf.predict(X_test)
        assert predictions.shape == (20,)
        assert np.all(np.isin(predictions, [0, 1, 2]))

        # Test predict_proba
        proba = clf.predict_proba(X_test)
        assert proba.shape == (20, 3)
        assert np.allclose(np.sum(proba, axis=1), 1.0)
        assert np.all((proba >= 0) & (proba <= 1))

    def test_not_fitted_error(self, binary_data):
        X, _ = binary_data
        clf = RandomForestTabPFNClassifier()

        with pytest.raises(NotFittedError):
            clf.predict(X)

        with pytest.raises(NotFittedError):
            clf.predict_proba(X)

    def test_input_validation(self, binary_data):
        X, y = binary_data
        clf = RandomForestTabPFNClassifier()

        # Test invalid X dimensions
        with pytest.raises(ValueError):
            clf.fit(X.reshape(-1), y)

        # Test mismatched X, y dimensions
        with pytest.raises(ValueError):
            clf.fit(X, y[:-1])

        # Test invalid y values for binary classification
        with pytest.raises(ValueError):
            clf.fit(X, y + 2)


class TestRandomForestTabPFNRegressor:
    def test_init(self):
        reg = RandomForestTabPFNRegressor()
        assert reg.max_time == 60
        assert reg.n_estimators == 10
        assert reg.random_state is None

    def test_regression(self, regression_data):
        X, y = regression_data
        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]

        reg = RandomForestTabPFNRegressor(random_state=42)
        reg.fit(X_train, y_train)

        # Test predict
        predictions = reg.predict(X_test)
        assert predictions.shape == (20,)
        assert predictions.dtype == np.float64

    def test_not_fitted_error(self, regression_data):
        X, _ = regression_data
        reg = RandomForestTabPFNRegressor()

        with pytest.raises(NotFittedError):
            reg.predict(X)

    def test_input_validation(self, regression_data):
        X, y = regression_data
        reg = RandomForestTabPFNRegressor()

        # Test invalid X dimensions
        with pytest.raises(ValueError):
            reg.fit(X.reshape(-1), y)

        # Test mismatched X, y dimensions
        with pytest.raises(ValueError):
            reg.fit(X, y[:-1])

    def test_max_time_validation(self, regression_data):
        X, y = regression_data

        # Test negative max_time
        with pytest.raises(ValueError):
            RandomForestTabPFNRegressor(max_time=-1)

        # Test zero max_time
        with pytest.raises(ValueError):
            RandomForestTabPFNRegressor(max_time=0)


def test_random_state_reproducibility():
    X, y = load_breast_cancer(return_X_y=True)
    X, y = X[:100], y[:100]
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]

    # Test classifier
    clf1 = RandomForestTabPFNClassifier(random_state=42)
    clf2 = RandomForestTabPFNClassifier(random_state=42)

    clf1.fit(X_train, y_train)
    clf2.fit(X_train, y_train)

    np.testing.assert_array_equal(clf1.predict(X_test), clf2.predict(X_test))
    np.testing.assert_array_almost_equal(
        clf1.predict_proba(X_test), clf2.predict_proba(X_test)
    )

    # Test regressor
    reg1 = RandomForestTabPFNRegressor(random_state=42)
    reg2 = RandomForestTabPFNRegressor(random_state=42)

    reg1.fit(X_train, y_train)
    reg2.fit(X_train, y_train)

    np.testing.assert_array_almost_equal(reg1.predict(X_test), reg2.predict(X_test))
