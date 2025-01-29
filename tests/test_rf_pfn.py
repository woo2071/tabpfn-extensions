import numpy as np
import pytest
from tabpfn_extensions.rf_pfn import (
    RandomForestTabPFNClassifier,
    DecisionTreeTabPFNClassifier,
    RandomForestTabPFNRegressor,
    DecisionTreeTabPFNRegressor,
)
from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor
from sklearn.utils.estimator_checks import check_estimator, parametrize_with_checks

@pytest.mark.parametrize("model_class", [
    (RandomForestTabPFNClassifier, TabPFNClassifier),
    (RandomForestTabPFNRegressor, TabPFNRegressor),
    (DecisionTreeTabPFNClassifier, TabPFNClassifier),
    (DecisionTreeTabPFNRegressor, TabPFNRegressor)
])
def test_sklearn_compatibility(model_class):
    """Test RandomForestTabPFN compatibility with different sklearn versions."""
    # Generate sample data
    rng = np.random.RandomState(42)
    X = rng.rand(100, 4)
    y = rng.randint(0, 2, 100)
    
    # Initialize classifier
    clf_class, clf_base_class = model_class
    clf_base = clf_base_class()
    kwargs = {
        "tabpfn": clf_base,
        **({"n_estimators": 2} if "RandomForest" in clf_class.__name__ else {}),  # Only add n_estimators for RandomForest
        "max_depth": 2
    }
    clf = clf_class(**kwargs)
    
    # This should work without errors on both sklearn <1.6 and >=1.6
    clf.fit(X, y)
    
    # Verify predictions work
    pred = clf.predict(X)
    assert pred.shape == (100,)
    if clf_base_class == TabPFNClassifier:
        assert np.all(np.isin(pred, [0, 1]))
    else:
        assert pred.dtype == np.float64

@pytest.mark.parametrize("model_class", [
    (RandomForestTabPFNClassifier, TabPFNClassifier),
    (RandomForestTabPFNRegressor, TabPFNRegressor),
    (DecisionTreeTabPFNClassifier, TabPFNClassifier),
    (DecisionTreeTabPFNRegressor, TabPFNRegressor)
])
def test_with_nan(model_class):
    """Test handling of float64 data with NaN values for sklearn < 1.6."""
    # Generate sample data with NaN values
    rng = np.random.RandomState(42)
    X = rng.rand(100, 4)
    X[0, 0] = np.nan  # Add a NaN value
    y = rng.randint(0, 2, 100)
    
    clf_class, clf_base_class = model_class
    clf_base = clf_base_class()
    kwargs = {
        "tabpfn": clf_base,
        **({"n_estimators": 2} if "RandomForest" in clf_class.__name__ else {}),
        "max_depth": 2
    }
    clf = clf_class(**kwargs)
    
    # This should work without errors
    clf.fit(X, y)
    
    # Test prediction with NaN values
    X_test = X.copy()
    pred = clf.predict(X_test)
    
    assert pred.shape == (100,)
    if clf_base_class == TabPFNClassifier:
        assert np.all(np.isin(pred, [0, 1]))
    else:
        assert pred.dtype == np.float64
