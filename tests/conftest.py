"""Test configuration for TabPFN Extensions.

This module provides pytest fixtures and configuration for testing TabPFN Extensions
with both the full TabPFN package and the TabPFN client. It detects which implementations
are available and configures test fixtures accordingly.

The module defines several pytest markers:
- requires_tabpfn: Tests that require the full TabPFN package
- requires_tabpfn_client: Tests that require the TabPFN client
- requires_any_tabpfn: Tests that require either implementation
- client_compatible: Tests that are compatible with TabPFN client
- slow: Tests that are computationally intensive (skipped in fast mode)

Key fixtures provided:
- tabpfn_classifier: Returns appropriate TabPFN classifier
- tabpfn_regressor: Returns appropriate TabPFN regressor
- synthetic_data: Returns synthetic dataset for testing

Helper functions:
- get_all_tabpfn_classifiers(): Returns all available classifier implementations
- get_all_tabpfn_regressors(): Returns all available regressor implementations
"""

from __future__ import annotations

import os

import numpy as np
import pytest

# Test configuration settings
FAST_TEST_MODE = (
    os.environ.get("FAST_TEST_MODE", "0") == "1"
)  # Skip slow tests by default
SMALL_TEST_SIZE = 20  # Number of samples to use in fast test mode
DEFAULT_TEST_SIZE = 100  # Number of samples to use in regular mode
DEFAULT_TIMEOUT = 60  # Default timeout in seconds

# Global variables to track TabPFN implementation availability
HAS_TABPFN = False  # Is full TabPFN package available?
HAS_TABPFN_CLIENT = False  # Is TabPFN client available?
HAS_ANY_TABPFN = False  # Is any implementation available?
TABPFN_SOURCE = (
    None  # Which implementation is preferred ("tabpfn", or "tabpfn_client")
)

# Check if TabPFN is available
try:
    import tabpfn
    from tabpfn import TabPFNClassifier, TabPFNRegressor

    HAS_TABPFN = True
    HAS_ANY_TABPFN = True
    if TABPFN_SOURCE is None:
        TABPFN_SOURCE = "tabpfn"
except ImportError:
    pass

# Check if TabPFN client is available
try:
    import tabpfn_client
    from tabpfn_client import (
        TabPFNClassifier as ClientTabPFNClassifier,
        TabPFNRegressor as ClientTabPFNRegressor,
    )

    HAS_TABPFN_CLIENT = True
    HAS_ANY_TABPFN = True
    if TABPFN_SOURCE is None:
        TABPFN_SOURCE = "tabpfn_client"
except ImportError:
    pass


# Add the option to run example files
def pytest_addoption(parser):
    """Add command-line options to pytest."""
    parser.addoption(
        "--run-examples",
        action="store_true",
        default=False,
        help="Run example files as part of the test suite",
    )


# Define markers for tests
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "requires_tabpfn: mark test to require TabPFN package"
    )
    config.addinivalue_line(
        "markers", "requires_tabpfn_client: mark test to require TabPFN client"
    )
    config.addinivalue_line(
        "markers", "requires_any_tabpfn: mark test to require any TabPFN implementation"
    )
    config.addinivalue_line(
        "markers", "client_compatible: mark test as compatible with TabPFN client"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow (will be skipped in fast test mode)"
    )
    config.addinivalue_line("markers", "example: mark test as testing an example file")


# Define a fixture to provide TabPFN classifier
@pytest.fixture
def tabpfn_classifier(request):
    """Return appropriate TabPFN classifier based on availability."""
    if not HAS_ANY_TABPFN:
        pytest.fail(
            "ERROR: No TabPFN implementation available. Install TabPFN client with 'pip install tabpfn-client', or TabPFN with 'pip install tabpfn'"
        )

    # Check if test is marked to require specific implementation
    requires_tabpfn = request.node.get_closest_marker("requires_tabpfn")
    requires_client = request.node.get_closest_marker("requires_tabpfn_client")

    if requires_tabpfn and not HAS_TABPFN:
        pytest.fail(
            "ERROR: Test requires TabPFN package but it's not installed. Run 'pip install tabpfn'"
        )

    if requires_client and not HAS_TABPFN_CLIENT:
        pytest.fail(
            "ERROR: Test requires TabPFN client but it's not installed. Run 'pip install tabpfn-client'"
        )

    # Return appropriate classifier
    if TABPFN_SOURCE == "tabpfn":
        return TabPFNClassifier()  # Let the model use the default device setting
    elif TABPFN_SOURCE == "tabpfn_client":
        return ClientTabPFNClassifier()

    pytest.fail(
        "ERROR: No TabPFN classifier available. Install TabPFN with 'pip install tabpfn', TabPFN client with 'pip install tabpfn-client''"
    )


# Define a fixture to provide TabPFN regressor
@pytest.fixture
def tabpfn_regressor(request):
    """Return appropriate TabPFN regressor based on availability."""
    if not HAS_ANY_TABPFN:
        pytest.fail(
            "ERROR: No TabPFN implementation available. Install TabPFN with 'pip install tabpfn', TabPFN client with 'pip install tabpfn-client''"
        )

    # Check if test is marked to require specific implementation
    requires_tabpfn = request.node.get_closest_marker("requires_tabpfn")
    requires_client = request.node.get_closest_marker("requires_tabpfn_client")

    if requires_tabpfn and not HAS_TABPFN:
        pytest.fail(
            "ERROR: Test requires TabPFN package but it's not installed. Run 'pip install tabpfn'"
        )

    if requires_client and not HAS_TABPFN_CLIENT:
        pytest.fail(
            "ERROR: Test requires TabPFN client but it's not installed. Run 'pip install tabpfn-client'"
        )

    # Return appropriate regressor
    if TABPFN_SOURCE == "tabpfn":
        return TabPFNRegressor()  # Let the model use the default device setting
    elif TABPFN_SOURCE == "tabpfn_client":
        return ClientTabPFNRegressor()

    pytest.fail(
        "ERROR: No TabPFN regressor available. Install TabPFN with 'pip install tabpfn', TabPFN client with 'pip install tabpfn-client''"
    )


# Skip or fail tests based on markers and available implementations
def pytest_runtest_setup(item):
    """Skip or fail tests based on markers and available implementations."""
    # Handle TabPFN availability markers
    if item.get_closest_marker("requires_tabpfn") and not HAS_TABPFN:
        pytest.fail(
            "ERROR: Test requires TabPFN package but it's not installed. Run 'pip install tabpfn'"
        )

    if item.get_closest_marker("requires_tabpfn_client") and not HAS_TABPFN_CLIENT:
        pytest.fail(
            "ERROR: Test requires TabPFN client but it's not installed. Run 'pip install tabpfn-client'"
        )

    if item.get_closest_marker("requires_any_tabpfn") and not HAS_ANY_TABPFN:
        pytest.fail(
            "ERROR: Test requires any TabPFN implementation but none is installed. Run 'pip install tabpfn', 'pip install tabpfn', or 'pip install tabpfn-client'"
        )

    # Client compatibility check
    if TABPFN_SOURCE == "tabpfn_client" and not item.get_closest_marker(
        "client_compatible"
    ):
        pytest.fail(
            "ERROR: Test is not compatible with TabPFN client but only TabPFN client is installed. Install TabPFN package with 'pip install tabpfn'"
        )

    # Skip slow tests in fast mode
    if FAST_TEST_MODE and item.get_closest_marker("slow"):
        pytest.skip("Skipping slow test in fast mode")

    # Skip examples that are too slow or resource-intensive for test environments
    if item.get_closest_marker("example"):
        try:
            # Access the name parameter directly from the item params
            params = item.callspec.params if hasattr(item, "callspec") else {}
            example_file = params.get("example_file", {})
            example_name = (
                example_file.get("name", "") if isinstance(example_file, dict) else ""
            )

            if example_name in [
                "large_datasets_example.py",
            ]:
                pytest.skip(
                    f"Example {example_name} skipped in test environment - corresponding functionality tested separately"
                )
        except (AttributeError, KeyError):
            # If we can't determine the name, continue anyway
            pass


# Generic data fixtures for testing
@pytest.fixture
def synthetic_data_classification():
    """Create synthetic classification data with controlled properties.

    Returns:
        tuple: (X_train, X_test, y_train, y_test) containing training and test data
    """
    n_samples = SMALL_TEST_SIZE if FAST_TEST_MODE else DEFAULT_TEST_SIZE

    # Use consistent seed for reproducibility
    rng = np.random.RandomState(42)

    # Generate features
    X = rng.rand(n_samples, 5)

    # Create a simple binary classification problem
    y = (X[:, 0] + X[:, 1] > 1).astype(int)

    # Split into train/test
    train_size = int(0.7 * n_samples)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    return X_train, X_test, y_train, y_test


@pytest.fixture
def synthetic_data_regression():
    """Create synthetic regression data with controlled properties.

    Returns:
        tuple: (X_train, X_test, y_train, y_test) containing training and test data
    """
    n_samples = SMALL_TEST_SIZE if FAST_TEST_MODE else DEFAULT_TEST_SIZE

    # Use consistent seed for reproducibility
    rng = np.random.RandomState(42)

    # Generate features
    X = rng.rand(n_samples, 5)

    # Create a simple regression problem
    y = 2 * X[:, 0] + 3 * X[:, 1] - 1.5 * X[:, 2] + rng.normal(0, 0.1, size=n_samples)

    # Split into train/test
    train_size = int(0.7 * n_samples)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    return X_train, X_test, y_train, y_test


# Helper function to get TabPFN classifiers from all available sources
def get_all_tabpfn_classifiers():
    """Return TabPFN classifiers from all available sources.

    Returns:
    -------
    Dict[str, object]
        A dictionary of classifiers, one for each available implementation source.
        Keys are the implementation names ("tabpfn", or "tabpfn_client").
        Values are classifier instances.
    """
    classifiers = {}

    if HAS_TABPFN:
        from tabpfn import TabPFNClassifier

        classifiers["tabpfn"] = TabPFNClassifier()  # Use default device

    if HAS_TABPFN_CLIENT:
        from tabpfn_client import TabPFNClassifier as ClientTabPFNClassifier

        classifiers["tabpfn_client"] = ClientTabPFNClassifier()

    return classifiers


# Helper function to get TabPFN regressors from all available sources
def get_all_tabpfn_regressors():
    """Return TabPFN regressors from all available sources.

    Returns:
    -------
    Dict[str, object]
        A dictionary of regressors, one for each available implementation source.
        Keys are the implementation names ("tabpfn", or "tabpfn_client").
        Values are regressor instances.
    """
    regressors = {}

    if HAS_TABPFN:
        from tabpfn import TabPFNRegressor

        regressors["tabpfn"] = TabPFNRegressor()  # Use default device

    if HAS_TABPFN_CLIENT:
        from tabpfn_client import TabPFNRegressor as ClientTabPFNRegressor

        regressors["tabpfn_client"] = ClientTabPFNRegressor()

    return regressors
