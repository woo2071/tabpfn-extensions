# TabPFN Extensions Testing Guide

This directory contains tests for TabPFN extensions. The tests are designed to work with both the TabPFN package and TabPFN-client backends.

## Test Organization

- Tests are organized by extension, with a separate file for each extension
- Common test utilities and fixtures are in `conftest.py`
- Special markers are used to indicate compatibility with different backends

## Running Tests

```bash
# Run all tests
pytest

# Run tests for a specific extension
pytest tests/test_classifier_as_regressor.py

# Run only tests compatible with TabPFN client
pytest -m client_compatible

# Run tests requiring full TabPFN package
pytest -m requires_tabpfn

# Run tests with verbose output
pytest -v
```

## Test Markers

We use the following markers to indicate test compatibility:

- `client_compatible`: Test works with TabPFN client
- `requires_tabpfn`: Test requires full TabPFN package
- `requires_any_tabpfn`: Test requires any TabPFN implementation
- `requires_tabpfn_client`: Test specifically requires TabPFN client

## Fixtures

The `conftest.py` file provides fixtures that automatically detect and use the appropriate TabPFN implementation:

- `tabpfn_classifier`: Provides a TabPFN classifier based on what's available
- `tabpfn_regressor`: Provides a TabPFN regressor based on what's available

## Helper Functions

- `get_all_tabpfn_classifiers()`: Returns all available TabPFN classifiers
- `get_all_tabpfn_regressors()`: Returns all available TabPFN regressors

## Writing Tests

When writing tests for new extensions:

1. Use the provided fixtures to access TabPFN models
2. Add appropriate markers for compatibility
3. Handle graceful skipping when needed features aren't available
4. Test with both TabPFN implementations when possible

Example:

```python
import pytest
import numpy as np

@pytest.mark.client_compatible  # Works with TabPFN client
def test_my_feature(tabpfn_classifier):
    # Test code using tabpfn_classifier
    assert tabpfn_classifier is not None
    
@pytest.mark.requires_tabpfn  # Requires full TabPFN package
def test_advanced_feature():
    # Test code requiring full TabPFN
    pass
```