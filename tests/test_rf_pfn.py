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


@pytest.mark.parametrize(
    "model_class",
    [
        (RandomForestTabPFNClassifier, TabPFNClassifier),
        (RandomForestTabPFNRegressor, TabPFNRegressor),
    ],
)
def test_comprehensive_nan_handling(model_class):
    """Test multiple patterns of NaN values in RF-PFN models.
    
    This test verifies that RF-PFN models can handle various patterns of missing values:
    1. Random NaNs throughout the dataset
    2. Entire columns with NaNs
    3. Specific samples with multiple NaNs
    4. NaNs in both training and test data
    """
    # Generate sample data - smaller in fast mode
    rng = np.random.RandomState(42)
    n_samples = SMALL_TEST_SIZE if FAST_TEST_MODE else DEFAULT_TEST_SIZE
    n_features = 5
    
    # Create base data
    X = rng.rand(n_samples, n_features)
    
    # Create appropriate target type
    if model_class[1] == TabPFNClassifier:
        y = rng.randint(0, 2, n_samples)
    else:
        y = rng.randn(n_samples)
        
    # Create different NaN patterns
    X_with_nans = X.copy()
    
    # Pattern 1: Random NaNs (5% of values)
    nan_mask = rng.random(X.shape) < 0.05
    X_with_nans[nan_mask] = np.nan
    
    # Pattern 2: One column completely NaN
    X_with_nans[:, 1] = np.nan
    
    # Pattern 3: First two samples have multiple NaNs
    X_with_nans[0, [0, 2, 3]] = np.nan
    X_with_nans[1, [1, 3, 4]] = np.nan
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_with_nans, y, test_size=0.33, random_state=42
    )
    
    # Initialize model with minimal settings for speed
    clf_class, clf_base_class = model_class
    
    # Use smaller forest in fast mode
    n_trees = 1 if FAST_TEST_MODE else 2
    
    # Create the model
    clf = clf_class(
        tabpfn=clf_base_class(),
        n_estimators=n_trees,
        max_depth=2,
        random_state=42
    )
    
    # Train with NaN data
    clf.fit(X_train, y_train)
    
    # Make predictions on data with NaNs
    y_pred = clf.predict(X_test)
    
    # Basic shape and type checks
    assert y_pred.shape == y_test.shape
    if clf_base_class == TabPFNClassifier:
        assert np.all(np.isin(y_pred, [0, 1]))
    else:
        assert y_pred.dtype == np.float64
        
    # Now create some extreme NaN patterns for prediction
    X_extreme_nans = X_test.copy()
    
    # Sample with all NaNs except one feature
    X_extreme_nans[0, :] = np.nan
    X_extreme_nans[0, 0] = 0.5
    
    # Sample with alternating NaNs
    X_extreme_nans[1, ::2] = np.nan
    
    # Try to predict
    y_extreme_pred = clf.predict(X_extreme_nans[:2])
    
    # Should still produce valid predictions
    assert y_extreme_pred.shape == (2,)
    if clf_base_class == TabPFNClassifier:
        assert np.all(np.isin(y_extreme_pred, [0, 1]))
    else:
        assert y_extreme_pred.dtype == np.float64


@pytest.mark.parametrize(
    "model_class",
    [
        (RandomForestTabPFNClassifier, TabPFNClassifier),
        (RandomForestTabPFNRegressor, TabPFNRegressor),
    ],
)
def test_nan_with_categorical_features(model_class):
    """Test RF-PFN models with NaN values in categorical features.
    
    This test focuses on the handling of missing values in categorical features,
    which require special handling compared to numerical features.
    """
    # Generate sample data - smaller in fast mode
    rng = np.random.RandomState(42)
    n_samples = SMALL_TEST_SIZE if FAST_TEST_MODE else DEFAULT_TEST_SIZE
    n_features = 5
    
    # Create base data with mix of numerical and categorical features
    X = np.zeros((n_samples, n_features))
    
    # Features 0, 2, 4 are numerical
    X[:, 0] = rng.randn(n_samples)
    X[:, 2] = rng.randn(n_samples)
    X[:, 4] = rng.randn(n_samples)
    
    # Features 1, 3 are categorical (integers)
    X[:, 1] = rng.randint(0, 3, n_samples)  # 3 categories
    X[:, 3] = rng.randint(0, 5, n_samples)  # 5 categories
    
    # Create appropriate target type
    if model_class[1] == TabPFNClassifier:
        y = rng.randint(0, 2, n_samples)
    else:
        y = rng.randn(n_samples)
    
    # Add NaNs to both categorical and numerical features
    X_with_nans = X.copy()
    
    # Random NaNs in numerical features
    for col in [0, 2, 4]:
        mask = rng.random(n_samples) < 0.1  # 10% missing
        X_with_nans[mask, col] = np.nan
    
    # Random NaNs in categorical features
    for col in [1, 3]:
        mask = rng.random(n_samples) < 0.1  # 10% missing
        X_with_nans[mask, col] = np.nan
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_with_nans, y, test_size=0.33, random_state=42
    )
    
    # Initialize model with categorical feature information
    clf_class, clf_base_class = model_class
    
    # Use smaller forest in fast mode
    n_trees = 1 if FAST_TEST_MODE else 2
    
    # Create model with categorical features specified
    clf = clf_class(
        tabpfn=clf_base_class(),
        n_estimators=n_trees,
        max_depth=2,
        categorical_features=[1, 3],  # Specify which features are categorical
        random_state=42
    )
    
    # Train with NaN data
    clf.fit(X_train, y_train)
    
    # Make predictions on data with NaNs
    y_pred = clf.predict(X_test)
    
    # Basic shape and type checks
    assert y_pred.shape == y_test.shape
    if clf_base_class == TabPFNClassifier:
        assert np.all(np.isin(y_pred, [0, 1]))
    else:
        assert y_pred.dtype == np.float64
        
    # Try with a sample that has NaNs only in categorical features
    X_cat_nans = X_test.copy()
    X_cat_nans[0, 1] = np.nan
    X_cat_nans[0, 3] = np.nan
    
    # And a sample with NaNs only in numerical features
    X_num_nans = X_test.copy()
    X_num_nans[1, 0] = np.nan
    X_num_nans[1, 2] = np.nan
    X_num_nans[1, 4] = np.nan
    
    # Combine for testing
    X_mixed_nans = np.vstack([X_cat_nans[0:1], X_num_nans[1:2]])
    
    # Predict
    y_special_pred = clf.predict(X_mixed_nans)
    
    # Should still work
    assert y_special_pred.shape == (2,)
    if clf_base_class == TabPFNClassifier:
        assert np.all(np.isin(y_special_pred, [0, 1]))
    else:
        assert y_special_pred.dtype == np.float64


def test_preprocess_data_nan_handling():
    """Test that the preprocess_data utility function properly handles NaN values."""
    from tabpfn_extensions.rf_pfn.utils import preprocess_data
    
    # Create data with various NaN patterns
    rng = np.random.RandomState(42)
    n_samples = 20
    n_features = 4
    
    # Create data with NaNs in different patterns
    X = rng.rand(n_samples, n_features)
    
    # Add NaNs
    # - First column: random NaNs
    X[rng.choice(n_samples, 5, replace=False), 0] = np.nan
    
    # - Second column: all NaNs
    X[:, 1] = np.nan
    
    # - Third column: no NaNs (control)
    
    # - Fourth column: first half NaNs
    X[:n_samples//2, 3] = np.nan
    
    # Test with nan_values=True (default behavior)
    X_processed_with_nan_handling = preprocess_data(
        X, 
        nan_values=True, 
        one_hot_encoding=False,
        normalization=False
    )
    
    # Check that there are no NaNs in the processed data
    assert not np.isnan(X_processed_with_nan_handling).any(), "NaNs should be handled by preprocess_data"
    
    # Check column 1 which was all NaNs - should be all zeros
    assert np.all(X_processed_with_nan_handling.iloc[:, 1] == 0), "Column of all NaNs should be replaced with zeros"
    
    # Test with nan_values=False (no NaN handling)
    X_processed_without_nan_handling = preprocess_data(
        X, 
        nan_values=False, 
        one_hot_encoding=False,
        normalization=False
    )
    
    # NaNs should be preserved
    assert np.isnan(X_processed_without_nan_handling.iloc[:, 1]).all(), "NaNs should be preserved with nan_values=False"
    
    # Check that column 0 has the same number of NaNs
    assert np.isnan(X_processed_without_nan_handling.iloc[:, 0]).sum() == 5
    
    # Test with categorical features and NaNs
    X_cat = X.copy()
    X_cat[:, 2] = rng.randint(0, 3, n_samples)  # Make column 2 categorical
    
    # Process with categorical features
    X_processed_with_cat = preprocess_data(
        X_cat,
        nan_values=True,
        one_hot_encoding=True,
        categorical_indices=[2]
    )
    
    # Should have more columns due to one-hot encoding
    assert X_processed_with_cat.shape[1] > X.shape[1], "One-hot encoding should increase feature count"
    
    # No NaNs should remain
    assert not np.isnan(X_processed_with_cat).any(), "NaNs should be handled in categorical features too"
