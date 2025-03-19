# TabPFN Extensions Development Guide

## Code Style & Standards
- **Type Hints**: Use Python type hints; import from `typing` and `typing_extensions`
- **Imports**: Always include `from __future__ import annotations` first
- **Formatting**: Max line length 88 characters; follow Google docstring style
- **Linting**: Code is linted with ruff (run `ruff check ./src` and `ruff format ./src`)
- **Error Handling**: Use descriptive error messages; include context in exceptions
- **Naming**: Use snake_case for functions/variables, PascalCase for classes
- **Copyright**: Include copyright headers in all source files (`Copyright (c) Prior Labs GmbH 2025. Licensed under the Apache License, Version 2.0`)

## Common Commands
```bash
# Installation
pip install -e ".[dev]"  # Development installation
pip install -e ".[dev,all]"  # All extensions

# Testing
python -m pytest tests/test_base_tabpfn.py -v  # Test specific file
python -m pytest tests/test_base_tabpfn.py::TestTabPFNClassifier::test_fit_predict -v  # Single test
FAST_TEST_MODE=1 python -m pytest  # Fast mode for all tests
python -m pytest -m client_compatible  # Only client-compatible tests
python -m pytest --backend=tabpfn  # Force TabPFN package backend
python -m pytest --backend=tabpfn_client  # Force TabPFN client backend

# Linting & Formatting
ruff check ./src  # Run linter
ruff format ./src  # Format code
mypy ./src  # Type checking
```

## Backend Options
- Use `tabpfn_extensions.utils.get_tabpfn_models()` for cross-backend compatibility
- Ensure tests work with both backends:
  1. **TabPFN Package**: Full implementation (`pip install tabpfn`)
  2. **TabPFN Client**: API client (`pip install tabpfn-client`)
- Use `@pytest.mark.client_compatible` for tests that work with both backends

## Test Data Utilities
The `DatasetGenerator` in `tests/utils.py` provides test data:
- Classification/regression datasets (numpy/pandas)
- Datasets with missing values (`generate_missing_values_dataset`)
- Datasets with text features (`generate_text_dataset`)
- Datasets with mixed types (`generate_mixed_types_dataset`)
- Datasets with correlations (`add_correlations`)

## Documentation
- Add docstrings to all public functions, classes, and methods
- Include examples in docstrings where helpful
- Update README.md when adding new functionality
