from __future__ import annotations

import os
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

try:
    from tabpfn_extensions.hpo import (
        TabPFNSearchSpace,
        TunedTabPFNClassifier,
        TunedTabPFNRegressor,
    )
    HAS_HPO_MODULES = True
except ImportError:
    HAS_HPO_MODULES = False

# Import common testing utilities from conftest

# HPO models can work with either TabPFN or client
pytestmark = [
    pytest.mark.requires_any_tabpfn,  # Requires any TabPFN implementation
    pytest.mark.client_compatible,    # Compatible with TabPFN client
    pytest.mark.skipif(not HAS_HPO_MODULES, reason="HPO modules not available"),
]


# Use fixtures from conftest.py
@pytest.fixture
def classification_data(synthetic_data_classification):
    """Create a simple classification dataset using common test fixture."""
    return synthetic_data_classification


@pytest.fixture
def regression_data(synthetic_data_regression):
    """Create a simple regression dataset using common test fixture."""
    return synthetic_data_regression


def test_tuned_classifier_basic(classification_data):
    """Test basic functionality of TunedTabPFNClassifier."""
    X_train, X_test, y_train, y_test = classification_data

    # Initialize with absolute minimal optimization (1 trial only)
    # Note: TunedTabPFNClassifier may use either max_evals or n_trials depending on the implementation
    try:
        model = TunedTabPFNClassifier(
            max_evals=1,  # Absolute minimum for testing
            device="cpu",
            n_validation_size=0.5,  # Faster training
            verbose=False,  # Reduce output
        )
    except TypeError:
        model = TunedTabPFNClassifier(
            n_trials=1,  # Absolute minimum for testing
            device="cpu",
            n_validation_size=0.5,  # Faster training
            verbose=False,  # Reduce output
        )

    # Fit the model
    model.fit(X_train, y_train)

    # Verify model fit properly
    assert hasattr(model, "best_params_")
    assert hasattr(model, "best_model_")

    # Make predictions
    y_pred = model.predict(X_test)
    assert y_pred.shape == y_test.shape

    # Get probabilities
    y_proba = model.predict_proba(X_test)
    assert y_proba.shape[0] == y_test.shape[0]
    assert np.all(np.isclose(y_proba.sum(axis=1), 1.0))


def test_tuned_regressor_basic(regression_data):
    """Test basic functionality of TunedTabPFNRegressor."""
    X_train, X_test, y_train, y_test = regression_data

    # Initialize with absolute minimal optimization (1 trial only)
    # Note: TunedTabPFNRegressor may use either max_evals or n_trials depending on the implementation
    try:
        model = TunedTabPFNRegressor(
            max_evals=1,  # Absolute minimum for testing
            device="cpu",
            n_validation_size=0.5,  # Faster training
            verbose=False,  # Reduce output
        )
    except TypeError:
        model = TunedTabPFNRegressor(
            n_trials=1,  # Absolute minimum for testing
            device="cpu",
            n_validation_size=0.5,  # Faster training
            verbose=False,  # Reduce output
        )

    # Fit the model
    model.fit(X_train, y_train)

    # Verify model fit properly
    assert hasattr(model, "best_params_")
    assert hasattr(model, "best_model_")

    # Make predictions
    y_pred = model.predict(X_test)
    assert y_pred.shape == y_test.shape
    assert np.all(np.isfinite(y_pred))


def test_search_space_customization(classification_data):
    """Test that custom search spaces work as expected."""
    X_train, X_test, y_train, y_test = classification_data

    # Create a TabPFNSearchSpace object for a minimal but valid search space
    custom_space = TabPFNSearchSpace.get_classifier_space(
        n_ensemble_range=(1, 2),  # Minimal values for testing
        temp_range=(0.5, 0.5),  # Single value to reduce search space
    )

    # Initialize with minimal optimization
    try:
        model = TunedTabPFNClassifier(
            max_evals=1,  # Minimum for testing
            search_space=custom_space,
            device="cpu",
            verbose=False,  # Reduce output
            n_validation_size=0.5,  # Faster training
        )
    except TypeError:
        # Fall back to n_trials if max_evals isn't supported
        try:
            model = TunedTabPFNClassifier(
                n_trials=1,  # Minimum for testing
                search_space=custom_space,
                device="cpu",
                verbose=False,  # Reduce output
                n_validation_size=0.5,  # Faster training
            )
        except TypeError:
            pytest.fail("Custom search space not supported in this version")
            return

    # Fit the model
    model.fit(X_train, y_train)

    # Verify model has best_params_
    assert hasattr(model, "best_params_")

    # Since the actual parameter format in best_params_ depends on hyperopt,
    # we'll check the model actually works instead of checking specific parameter values

    # Make predictions to verify model works with custom search space
    y_pred = model.predict(X_test)
    assert y_pred.shape == y_test.shape

    # If there are probabilities, check they're valid
    y_proba = model.predict_proba(X_test)
    assert y_proba.shape[0] == y_test.shape[0]
    assert np.all(np.isclose(y_proba.sum(axis=1), 1.0))


def test_objective_function(classification_data):
    """Test that different objective functions can be used."""
    X_train, X_test, y_train, y_test = classification_data

    # Define a custom objective function
    def custom_objective(model, X, y):
        y_pred = model.predict(X)
        return -accuracy_score(y, y_pred)  # Negative because hyperopt minimizes

    # Initialize with custom objective - use absolute minimum trial count
    # Note: Parameter names may be different depending on TabPFN version
    try:
        model = TunedTabPFNClassifier(
            max_evals=1,  # Minimum for testing
            objective_fn=custom_objective,
            device="cpu",
            verbose=False,  # Reduce output
            n_validation_size=0.5,  # Faster training
        )
    except TypeError:
        try:
            model = TunedTabPFNClassifier(
                n_trials=1,  # Minimum for testing
                objective_fn=custom_objective,
                device="cpu",
                verbose=False,  # Reduce output
                n_validation_size=0.5,  # Faster training
            )
        except TypeError:
            # If custom objective is named differently or not supported, fail the test
            pytest.fail("Custom objective function not supported in current TabPFN version")
            return

    # Fit the model
    model.fit(X_train, y_train)

    # Should complete without errors
    assert hasattr(model, "best_model_")


def test_search_space_class():
    """Test the TabPFNSearchSpace class."""
    try:
        # Test classifier search space
        clf_space = TabPFNSearchSpace.get_classifier_space()
        assert isinstance(clf_space, dict)

        # Check for ensemble parameter (name may vary by TabPFN version)
        param_name = next((p for p in ["n_estimators", "N_ensemble_configurations"] if p in clf_space), None)
        assert param_name is not None, "Ensemble parameter not found in search space"

        # Test regressor search space
        reg_space = TabPFNSearchSpace.get_regressor_space()
        assert isinstance(reg_space, dict)

        # Test custom parameter ranges (parameter name may vary by TabPFN version)
        custom_range = TabPFNSearchSpace.get_classifier_space(
            n_ensemble_range=(5, 15),
            temp_range=(0.1, 0.5),
        )

        param_name = next((p for p in ["n_estimators", "N_ensemble_configurations"] if p in custom_range), None)
        assert param_name is not None, "Ensemble parameter not found in custom search space"

        if param_name in custom_range:
            assert custom_range[param_name][0] >= 5
            assert custom_range[param_name][-1] <= 15
    except (ImportError, AttributeError) as e:
        pytest.skip(f"TabPFNSearchSpace not properly initialized: {e}")


def test_sklearn_compatibility(classification_data, regression_data):
    """Test that tuned models follow sklearn conventions."""
    X_train, X_test, y_train, y_test = classification_data
    X_reg_train, X_reg_test, y_reg_train, y_reg_test = regression_data

    # Test classifier with appropriate parameter name - use absolute minimum trials
    try:
        clf = TunedTabPFNClassifier(
            max_evals=1,  # Minimum for testing
            device="cpu",
            verbose=False,  # Reduce output
            n_validation_size=0.5,  # Faster training
        )
        param_key = "max_evals"
    except TypeError:
        clf = TunedTabPFNClassifier(
            n_trials=1,  # Minimum for testing
            device="cpu",
            verbose=False,  # Reduce output
            n_validation_size=0.5,  # Faster training
        )
        param_key = "n_trials"

    # Test sklearn API compatibility
    params = clf.get_params()
    assert param_key in params

    # Fit and check score method
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    assert 0 <= score <= 1

    # Test regressor with appropriate parameter name - use absolute minimum trials
    try:
        reg = TunedTabPFNRegressor(
            max_evals=1,  # Minimum for testing
            device="cpu",
            verbose=False,  # Reduce output
            n_validation_size=0.5,  # Faster training
        )
    except TypeError:
        reg = TunedTabPFNRegressor(
            n_trials=1,  # Minimum for testing
            device="cpu",
            verbose=False,  # Reduce output
            n_validation_size=0.5,  # Faster training
        )

    # Fit and check score method
    reg.fit(X_reg_train, y_reg_train)
    reg_score = reg.score(X_reg_test, y_reg_test)
    # R^2 score for a simple model might be negative or low
    assert isinstance(reg_score, float)


def test_tuned_vs_default_classifier():
    """Test that tuned model performs at least as well as a default model."""
    # Create dataset with clear pattern - 3 classes with 3 features
    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        n_classes=3,
        random_state=42,
    )

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42,
    )

    # Try both possible parameter names for number of trials
    try:
        # Create tuned classifier with 2 trials (minimal but better than 1)
        tuned_clf = TunedTabPFNClassifier(
            max_evals=2,
            device="cpu",
            verbose=False,
            n_validation_size=0.5,
        )
    except TypeError:
        # Create tuned classifier with 2 trials (minimal but better than 1)
        tuned_clf = TunedTabPFNClassifier(
            n_trials=2,
            device="cpu",
            verbose=False,
            n_validation_size=0.5,
        )

    # Create default classifier for comparison
    try:
        from tabpfn import TabPFNClassifier
        default_clf = TabPFNClassifier()  # Let device default to auto
    except ImportError:
        try:
            from tabpfn_client import TabPFNClassifier
            default_clf = TabPFNClassifier()
        except ImportError:
            from tabpfn_extensions import TabPFNClassifier
            default_clf = TabPFNClassifier()  # Let device default to auto

    # Fit both models
    tuned_clf.fit(X_train, y_train)
    default_clf.fit(X_train, y_train)

    # Get predictions
    tuned_preds = tuned_clf.predict(X_test)
    default_preds = default_clf.predict(X_test)

    # Compare accuracy - tuned should be at least as good as default
    # Since we're using just 2 trials, use a relaxed condition
    tuned_acc = accuracy_score(y_test, tuned_preds)
    default_acc = accuracy_score(y_test, default_preds)

    # Tuned model should be at least 70% as good as default in fast test mode,
    # or 80% as good in regular mode (with only 2 trials, we can't expect it to always be better)
    min_ratio = 0.7 if os.environ.get("FAST_TEST_MODE", "0") == "1" else 0.8
    assert tuned_acc >= min_ratio * default_acc

    # Check multi-class probabilities
    tuned_proba = tuned_clf.predict_proba(X_test)
    assert tuned_proba.shape == (len(X_test), 3)  # 3 classes
    assert np.allclose(tuned_proba.sum(axis=1), 1.0)


def test_regression_metrics(regression_data):
    """Test different regression metrics as shown in the example."""
    X_train, X_test, y_train, y_test = regression_data

    # Try both possible parameter names for number of trials
    try:
        # Create minimal regressor for testing
        reg = TunedTabPFNRegressor(
            max_evals=1,
            device="cpu",
            verbose=False,
            n_validation_size=0.5,
        )
    except TypeError:
        # Create minimal regressor for testing
        reg = TunedTabPFNRegressor(
            n_trials=1,
            device="cpu",
            verbose=False,
            n_validation_size=0.5,
        )

    # Fit the model
    reg.fit(X_train, y_train)

    # Make predictions
    y_pred = reg.predict(X_test)

    # Test different metrics like in the example
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Simple type checks - values will depend on data
    assert isinstance(mse, float)
    assert isinstance(mae, float)
    assert isinstance(r2, float)

    # MSE should be positive
    assert mse >= 0
    # MAE should be positive
    assert mae >= 0