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
TABPFN_SOURCE = None  # Which implementation is preferred ("tabpfn", or "tabpfn_client")

# This will be set in pytest_configure based on command line option
FORCE_BACKEND = None  # Force specific backend if set

# Check if TabPFN is available
try:
    # Using importlib.util.find_spec as recommended by ruff
    import importlib.util
    
    # For debugging
    import sys
    print("Python path:", sys.path)
    print("Looking for TabPFN...")
    
    tabpfn_spec = importlib.util.find_spec("tabpfn")
    print("TabPFN spec:", tabpfn_spec)
    
    if tabpfn_spec is not None:
        print("Found tabpfn package")
        from tabpfn import TabPFNClassifier, TabPFNRegressor
        print("Imported TabPFNClassifier, TabPFNRegressor")
        
        HAS_TABPFN = True
        HAS_ANY_TABPFN = True
        if TABPFN_SOURCE is None:
            TABPFN_SOURCE = "tabpfn"
except ImportError as e:
    print("ImportError when importing tabpfn:", e)
except Exception as e:
    print("Exception when importing tabpfn:", e)

# Check if TabPFN client is available
try:
    # Using importlib.util.find_spec as recommended by ruff
    print("Looking for TabPFN client...")
    tabpfn_client_spec = importlib.util.find_spec("tabpfn_client")
    print("TabPFN client spec:", tabpfn_client_spec)
    
    if tabpfn_client_spec is not None:
        print("Found tabpfn_client package")
        from tabpfn_client import TabPFNClassifier as ClientTabPFNClassifier
        from tabpfn_client import TabPFNRegressor as ClientTabPFNRegressor
        print("Imported ClientTabPFNClassifier, ClientTabPFNRegressor")
        
        HAS_TABPFN_CLIENT = True
        HAS_ANY_TABPFN = True
        if TABPFN_SOURCE is None:
            TABPFN_SOURCE = "tabpfn_client"
except ImportError as e:
    print("ImportError when importing tabpfn_client:", e)
except Exception as e:
    print("Exception when importing tabpfn_client:", e)


# Add command-line options
def pytest_addoption(parser):
    """Add command-line options to pytest."""
    parser.addoption(
        "--run-examples",
        action="store_true",
        default=False,
        help="Run example files as part of the test suite",
    )
    parser.addoption(
        "--backend",
        action="store",
        default="all",
        help="Specify which TabPFN backend to use: 'tabpfn', 'tabpfn_client', or 'all' (use both if available)",
    )


# Define markers for tests
def pytest_configure(config):
    """Configure pytest markers and handle backend selection."""
    global TABPFN_SOURCE, FORCE_BACKEND
    
    # Register test markers
    config.addinivalue_line(
        "markers",
        "requires_tabpfn: mark test to require TabPFN package",
    )
    config.addinivalue_line(
        "markers",
        "requires_tabpfn_client: mark test to require TabPFN client",
    )
    config.addinivalue_line(
        "markers",
        "client_compatible: mark test as compatible with TabPFN client",
    )
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow (will be skipped in fast test mode)",
    )
    config.addinivalue_line("markers", "example: mark test as testing an example file")
    
    # Handle backend selection from command line
    backend_option = config.getoption("--backend")
    
    if backend_option == "tabpfn" and HAS_TABPFN:
        print("Forcing TabPFN package backend as requested")
        TABPFN_SOURCE = "tabpfn"
        FORCE_BACKEND = "tabpfn"
    elif backend_option == "tabpfn_client" and HAS_TABPFN_CLIENT:
        print("Forcing TabPFN client backend as requested")
        TABPFN_SOURCE = "tabpfn_client"
        FORCE_BACKEND = "tabpfn_client"
    elif backend_option == "auto":
        # Auto mode uses what's available (no change to previous behavior)
        pass
    elif backend_option == "tabpfn" and not HAS_TABPFN:
        print("WARNING: Requested TabPFN package backend but it's not available")
    elif backend_option == "tabpfn_client" and not HAS_TABPFN_CLIENT:
        print("WARNING: Requested TabPFN client backend but it's not available")
        
    # Print status information
    print(f"Test configuration:")
    print(f"  Backend: {TABPFN_SOURCE or 'none'}")
    print(f"  TabPFN package available: {HAS_TABPFN}")
    print(f"  TabPFN client available: {HAS_TABPFN_CLIENT}")
    print(f"  Fast test mode: {FAST_TEST_MODE}")
    print(f"  Default timeout: {DEFAULT_TIMEOUT} seconds")


# Helper to determine which backends to test
def get_backends_to_test(config):
    """Determine which backends to test based on command line options."""
    backend_option = config.getoption("--backend")
    
    if backend_option == "tabpfn" and HAS_TABPFN:
        return ["tabpfn"]
    elif backend_option == "tabpfn_client" and HAS_TABPFN_CLIENT:
        return ["tabpfn_client"]
    else:
        # Include all available backends
        backends = []
        if HAS_TABPFN:
            backends.append("tabpfn")
        if HAS_TABPFN_CLIENT:
            backends.append("tabpfn_client")
        return backends

# Define a parametrization hook to test with both backends
def pytest_generate_tests(metafunc):
    """Generate tests for each backend."""
    if "backend" in metafunc.fixturenames:
        backends = get_backends_to_test(metafunc.config)
        
        # Skip the whole test if no backends are available
        if not backends:
            pytest.skip("No TabPFN backends available")
            
        metafunc.parametrize("backend", backends, scope="function", 
                            ids=[f"backend={b}" for b in backends])

# Define a fixture to evaluate if a test should run for a given backend
@pytest.fixture
def backend(request):
    """Fixture that provides each available backend."""
    backend_name = request.param
    
    # Skip if the test requires a specific backend
    requires_tabpfn = request.node.get_closest_marker("requires_tabpfn")
    requires_client = request.node.get_closest_marker("requires_tabpfn_client")
    
    if requires_tabpfn and backend_name != "tabpfn":
        pytest.skip("Test requires TabPFN package")
    if requires_client and backend_name != "tabpfn_client":
        pytest.skip("Test requires TabPFN client")
        
    # Skip if the test is not compatible with client and we're using client
    if backend_name == "tabpfn_client" and not request.node.get_closest_marker("client_compatible"):
        pytest.skip("Test is not compatible with TabPFN client")
        
    return backend_name

# Define a fixture to provide TabPFN classifier
@pytest.fixture
def tabpfn_classifier(backend):
    """Return TabPFN classifier for the current backend."""
    if backend == "tabpfn":
        return TabPFNClassifier()  # Let the model use the default device setting
    elif backend == "tabpfn_client":
        return ClientTabPFNClassifier()
    else:
        pytest.fail(f"Unknown backend: {backend}")


# Define a fixture to provide TabPFN regressor
@pytest.fixture
def tabpfn_regressor(backend):
    """Return TabPFN regressor for the current backend."""
    if backend == "tabpfn":
        return TabPFNRegressor()  # Let the model use the default device setting
    elif backend == "tabpfn_client":
        return ClientTabPFNRegressor()
    else:
        pytest.fail(f"Unknown backend: {backend}")


# Fixtures for standard datasets

@pytest.fixture
def dataset_generator():
    """Create a dataset generator with a fixed seed."""
    # Use relative import to avoid 'tests' module not found issue
    import sys, os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from utils import DatasetGenerator
    return DatasetGenerator(seed=42)

@pytest.fixture
def classification_data(dataset_generator):
    """Return a simple classification dataset with 2 classes."""
    test_size = SMALL_TEST_SIZE if FAST_TEST_MODE else DEFAULT_TEST_SIZE
    X, y = dataset_generator.generate_classification_data(
        n_samples=test_size, n_features=5, n_classes=2
    )
    return X, y

@pytest.fixture
def multiclass_data(dataset_generator):
    """Return a classification dataset with 3+ classes."""
    test_size = SMALL_TEST_SIZE if FAST_TEST_MODE else DEFAULT_TEST_SIZE
    X, y = dataset_generator.generate_classification_data(
        n_samples=test_size, n_features=5, n_classes=5
    )
    return X, y

@pytest.fixture
def regression_data(dataset_generator):
    """Return a simple regression dataset."""
    test_size = SMALL_TEST_SIZE if FAST_TEST_MODE else DEFAULT_TEST_SIZE
    X, y = dataset_generator.generate_regression_data(
        n_samples=test_size, n_features=5
    )
    return X, y

@pytest.fixture
def pandas_classification_data(dataset_generator):
    """Return a classification dataset as pandas DataFrame."""
    test_size = SMALL_TEST_SIZE if FAST_TEST_MODE else DEFAULT_TEST_SIZE
    X, y = dataset_generator.generate_classification_data(
        n_samples=test_size, n_features=5, n_classes=2, as_pandas=True
    )
    return X, y

@pytest.fixture
def pandas_regression_data(dataset_generator):
    """Return a regression dataset as pandas DataFrame."""
    test_size = SMALL_TEST_SIZE if FAST_TEST_MODE else DEFAULT_TEST_SIZE
    X, y = dataset_generator.generate_regression_data(
        n_samples=test_size, n_features=5, as_pandas=True
    )
    return X, y

@pytest.fixture
def mixed_type_data(dataset_generator):
    """Return a dataset with mixed numerical and categorical features."""
    test_size = SMALL_TEST_SIZE if FAST_TEST_MODE else DEFAULT_TEST_SIZE
    return dataset_generator.dataset_with_mixed_types(
        n_samples=test_size, n_numerical=3, n_categorical=2
    )

@pytest.fixture
def data_with_missing_values(dataset_generator):
    """Return a dataset with missing values."""
    test_size = SMALL_TEST_SIZE if FAST_TEST_MODE else DEFAULT_TEST_SIZE
    return dataset_generator.dataset_with_missing_values(
        n_samples=test_size, n_features=5
    )

@pytest.fixture
def data_with_outliers(dataset_generator):
    """Return a dataset with outliers."""
    test_size = SMALL_TEST_SIZE if FAST_TEST_MODE else DEFAULT_TEST_SIZE
    return dataset_generator.dataset_with_outliers(
        n_samples=test_size, n_features=5
    )


# Skip or fail tests based on markers and available implementations
def pytest_runtest_setup(item):
    """Skip or fail tests based on markers and available implementations."""
    # Handle TabPFN availability markers
    if item.get_closest_marker("requires_tabpfn") and not HAS_TABPFN:
        pytest.fail(
            "ERROR: Test requires TabPFN package but it's not installed. Run 'pip install tabpfn'",
        )

    if item.get_closest_marker("requires_tabpfn_client") and not HAS_TABPFN_CLIENT:
        pytest.fail(
            "ERROR: Test requires TabPFN client but it's not installed. Run 'pip install tabpfn-client'",
        )

    if item.get_closest_marker("requires_any_tabpfn") and not HAS_ANY_TABPFN:
        pytest.fail(
            "ERROR: Test requires any TabPFN implementation but none is installed. Run 'pip install tabpfn', 'pip install tabpfn', or 'pip install tabpfn-client'",
        )

    # Client compatibility check
    if TABPFN_SOURCE == "tabpfn_client" and not item.get_closest_marker(
        "client_compatible",
    ):
        pytest.fail(
            "ERROR: Test is not compatible with TabPFN client but only TabPFN client is installed. Install TabPFN package with 'pip install tabpfn'",
        )

    # Skip slow tests in fast mode
    if FAST_TEST_MODE and item.get_closest_marker("slow"):
        pytest.skip("Skipping slow test in fast mode")