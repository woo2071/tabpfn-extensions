# TabPFN Extensions Testing Guide

## Overview
TabPFN Extensions is a collection of extensions and utilities for the TabPFN foundation model for tabular data. This guide focuses on the testing approach to ensure all extensions work correctly with both TabPFN implementations.

## Backend Options
Tests should work with both backends:

1. **TabPFN Package** - Full PyTorch implementation:
   ```bash
   pip install tabpfn
   ```

2. **TabPFN Client** - Lightweight API client:
   ```bash
   pip install tabpfn-client
   ```

## Test Philosophy

1. **Simplicity**: Write simple, focused tests that verify core functionality
2. **Reusability**: Create a reusable test framework with base classes and fixtures
3. **Coverage**: Test all extensions with both backends where possible
4. **Data Variety**: Test with numpy arrays, pandas DataFrames, missing values, and text features

## Important Testing Notes

1. **Always Use Conda Environment**: Tests must run from the conda environment where TabPFN is installed
   - Use `python -m pytest` instead of just `pytest` to ensure the right Python version
   - Otherwise, tests will fail to detect TabPFN even if it's installed

2. **Use Timeouts**: Some tests can run for a long time
   - Add `--timeout=30` to limit test runtime
   - Install with `pip install pytest-timeout`

3. **Run Tests One by One**: When debugging issues, run individual tests
   - Test one file: `python -m pytest tests/test_base_tabpfn.py`
   - Test one class: `python -m pytest tests/test_base_tabpfn.py::TestTabPFNClassifier`
   - Test one method: `python -m pytest tests/test_base_tabpfn.py::TestTabPFNClassifier::test_fit_predict`

## Testing Status

The tests have been successfully set up and run for:

1. **Base TabPFN tests**: Core functionality of TabPFN is tested
   - Classification and regression work properly
   - All tests pass when using the correct Python environment

2. **TabPFN RF extension tests**: The RandomForest extension is tested
   - Core functionality works (fit, predict, multiclass, etc.)
   - Some advanced features (pandas DataFrames, text features) are skipped as they require modifications to the RF extension

3. **Known Limitations**:
   - Must use the correct Python interpreter (`python -m pytest` instead of `pytest`)
   - Some slow tests are skipped in fast mode
   - TabPFN RF has some limitations with pandas DataFrames

## Getting Started with Testing

```bash
# Quick test run (IMPORTANT: always use Python from conda environment)
conda activate tabpfn-env
FAST_TEST_MODE=1 python -m pytest tests/test_base_tabpfn.py -v

# Test specific module with timeout
conda activate tabpfn-env
FAST_TEST_MODE=1 python -m pytest tests/test_rf_pfn.py -v --timeout=30

# Run a single test
conda activate tabpfn-env
FAST_TEST_MODE=1 python -m pytest tests/test_base_tabpfn.py::TestTabPFNClassifier::test_fit_predict -v

# Run all tests
conda activate tabpfn-env
FAST_TEST_MODE=1 python -m pytest tests/ -v

# Run only client-compatible tests
conda activate tabpfn-env
python -m pytest -m client_compatible
```

## Test Organization

1. **Base Tests (`test_base_tabpfn.py`)**:
   - Tests basic TabPFN functionality
   - Includes reusable `BaseClassifierTests` and `BaseRegressorTests` classes
   - Tests various data types and scikit-learn compatibility

2. **Extension Tests**:
   - Each extension has its own test file
   - Tests should inherit from base test classes when appropriate
   - Tests should be marked with compatibility markers

3. **Common Fixtures**:
   - Defined in `conftest.py`
   - Provide TabPFN models and datasets
   - Handle backend-specific configuration

## Writing New Tests

1. **Inherit from Base Classes**:
   ```python
   from test_base_tabpfn import BaseClassifierTests

   class TestMyExtension(BaseClassifierTests):
       @pytest.fixture
       def estimator(self, tabpfn_classifier):
           return MyExtensionModel(base_model=tabpfn_classifier)
   ```

2. **Use Data Generator**:
   ```python
   def test_my_feature(self, estimator, dataset_generator):
       X, y = dataset_generator.generate_classification_data(
           n_samples=30 if FAST_TEST_MODE else 60,
           n_features=5
       )
       # Test implementation...
   ```

3. **Add Compatibility Markers**:
   ```python
   @pytest.mark.client_compatible  # Works with TabPFN client
   @pytest.mark.slow  # Skip in FAST_TEST_MODE
   def test_complex_feature(self, estimator):
       # Test implementation...
   ```

## Test Data Types

The DatasetGenerator in `tests/utils.py` provides:

1. **Basic datasets**: Random numpy arrays or pandas DataFrames
2. **Correlated features**: Datasets with linear and non-linear relationships
3. **Missing values**: Datasets with NaN values + imputed versions
4. **Text features**: Datasets with categorical text columns + encoded versions
5. **Mixed types**: Datasets with numerical and categorical features

## Running Specific Test Types

```bash
# Only backend compatibility tests
pytest -m backend_compatibility

# Skip slow tests
FAST_TEST_MODE=1 pytest

# Test only specific extensions
pytest tests/test_post_hoc_ensembles.py tests/test_interpretability.py

# Test with a specific backend
pytest --backend=tabpfn        # Only use TabPFN package
pytest --backend=tabpfn_client # Only use TabPFN client
pytest --backend=all           # Use all available backends (default)
```

## Backend Testing

Tests will automatically run with both backends if they are available:

1. Tests marked with `@pytest.mark.client_compatible` will run with both the TabPFN package and the TabPFN client
2. Tests marked with `@pytest.mark.requires_tabpfn` will only run with the TabPFN package
3. Tests marked with `@pytest.mark.requires_tabpfn_client` will only run with the TabPFN client

This ensures comprehensive testing across both implementations. The test output will show which backend was used for each test run.
