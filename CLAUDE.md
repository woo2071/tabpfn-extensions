# TabPFN Extensions Development Guide

## Overview
TabPFN Extensions is a collection of extensions and utilities for the TabPFN foundation model for tabular data. Extensions include post-hoc ensembles, interpretability tools, hyperparameter optimization, and more.

## Installation Options

### For Users
```bash
# Install with all extensions
pip install "tabpfn-extensions[all]"

# Install with specific extensions
pip install "tabpfn-extensions[interpretability,hpo]"

# Development installation
pip install -e ".[dev,all]"
```

### Backend Options
TabPFN Extensions supports multiple backends (in priority order):

1. **TabPFN v2** - Next generation model with improved performance:
   ```bash
   pip install tabpfnv2
   ```

2. **TabPFN Package** - Full PyTorch implementation for local inference:
   ```bash
   pip install tabpfn
   ```

3. **TabPFN Client** - Lightweight API client for cloud-based inference:
   ```bash
   pip install tabpfn-client
   ```

#### Controlling Backend Selection
You can control which backend is used through environment variables:

```bash
# Force using only local TabPFN implementations (not client)
export USE_TABPFN_LOCAL=true  # Default: true

# Enable debug output when importing TabPFN
export TABPFN_DEBUG=true      # Default: false
```

## Build & Test Commands
### Essential For Development
- Install minimal dev dependencies: `pip install -e ".[dev]"`
- Quick test run: `FAST_TEST_MODE=1 pytest tests/test_your_module.py -v`
- Extra debug mode: `TABPFN_DEBUG=1 FAST_TEST_MODE=1 pytest tests/test_your_module.py -v`
- Full test of one module: `pytest tests/test_your_module.py -v`

### More Comprehensive Tests
- Run all tests: `pytest tests/`
- Run tests in fast mode: `FAST_TEST_MODE=1 pytest tests/`
- Run client-compatible tests: `pytest -m client_compatible`
- Run tests requiring full TabPFN: `pytest -m requires_tabpfn`
- Run tests requiring TabPFN v2: `pytest -m requires_tabpfnv2`

### Code Quality (Optional)
- Lint code: `ruff check src/tabpfn_extensions/your_module/`
- Type check: `mypy src/tabpfn_extensions/your_module/`
- Verify backend compatibility: `pytest -m backend_compatibility`

## Code Style Guidelines
- Use Google docstring style for documentation
- Line length: 88 characters
- Required imports: `from __future__ import annotations`
- Use type annotations for all function parameters and return values
- Follow isort conventions: [known-first-party="tabpfn", known-third-party="sklearn"]
- File organization: extensions in `src/tabpfn_extensions/`, examples in `examples/`, tests in `tests/`
- Use pytest fixtures and parametrize for efficient testing
- Error handling: Raise specific exceptions with descriptive messages
- Avoid hardcoding device parameters (use `auto` or configuration-based defaults)
- Support multiple input formats (NumPy arrays, pandas DataFrames, PyTorch tensors)

## Project Structure
Each extension should be in its own subpackage with:
- Implementation in `src/tabpfn_extensions/your_package/`
- Examples in `examples/your_package/`
- Tests in `tests/test_your_package.py` or `tests/your_package/`
- README.md documenting the extension's purpose and usage

## Extension Compatibility
When developing extensions, ensure compatibility with all TabPFN implementations:

1. **Import TabPFN from the central utility**:
   ```python
   # Import TabPFN models from the central utility that handles backend selection
   from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor
   
   # The utility automatically tries to import TabPFN in this order:
   # 1. TabPFN v2 (if available)
   # 2. Standard TabPFN package (if USE_TABPFN_LOCAL=true)
   # 3. TabPFN client
   
   # For debugging, you can enable verbose output:
   # import os
   # os.environ["TABPFN_DEBUG"] = "true"
   ```

2. **Use common parameters and avoid backend-specific features**:
   - Avoid using parameters that only exist in one implementation
   - Use default parameter values when possible (especially for device)
   - Document any backend-specific parameters clearly
   - Create abstraction layers for backend-specific functionality

3. **Test with all backends**:
   - Use the test markers `client_compatible`, `requires_tabpfn`, and `requires_tabpfnv2`
   - Use the fixtures provided in conftest.py for TabPFN instances
   - Create parametrized tests that run with each backend

## Quality Metrics and Best Practices

### Code Quality Metrics
- **Backend Compatibility**: >95% of features should work with all backends
- **Test Coverage**: Maintain >80% test coverage for all extensions
- **Documentation**: All public APIs must have complete docstrings
- **Type Annotations**: 100% of public functions should have type annotations
- **CI Pass Rate**: All tests must pass in CI pipeline before merging

### Best Practices
1. **Device Management**:
   - Never hardcode `device="cpu"` in production code
   - Use auto-detection with `device="auto"` (the default value)
   - Always use the `get_device()` utility from `tabpfn_extensions.utils`
   - Handle device parameters consistently across functions
   - Example usage:
     ```python
     from tabpfn_extensions.utils import get_device

     # In your extension's initialization:
     def __init__(self, device="auto", ...):
         self.device = device

     # When creating TabPFN models:
     model = TabPFNClassifier(device=get_device(self.device), ...)
     ```

2. **Input Validation**:
   - Support multiple input formats (NumPy arrays, pandas DataFrames, PyTorch tensors)
   - Validate inputs early and provide clear error messages
   - Implement consistent type conversion utilities

3. **Error Handling**:
   - Use specific exception types for different error conditions
   - Include descriptive error messages that suggest solutions
   - Add context to exceptions raised from lower-level functions

4. **Documentation**:
   - Document parameters, return values, exceptions, and examples
   - Note any backend-specific behavior or limitations
   - Keep READMEs updated with latest usage information

5. **Testing**:
   - Write tests for each backend separately where necessary
   - Use small, synthetic datasets for test speed
   - Test edge cases and error conditions explicitly

6. **Pull Requests**:
   - **DO NOT** add "Done with Claude" or AI attribution to PRs
   - Include basic PR description explaining changes
   - Reference related issues in PR descriptions
   - Respond to code review comments
   - Test your specific module before submitting: `pytest tests/test_your_module.py`
   - For quicker testing use: `FAST_TEST_MODE=1 pytest tests/test_your_module.py`

## Long-Term Code Quality Improvement Plan

### Phase 1: Foundation (0-3 months)
- ✅ Complete device parameter handling standardization with `get_device()` utility
- Ensure all extensions support multiple input formats
- Add missing type annotations to all public APIs
- Implement comprehensive CI workflows for all quality checks
- Create consistent documentation templates

### Phase 2: Enhanced Compatibility (3-6 months)
- Develop comprehensive backend compatibility test suite
- Refactor backend-specific code behind abstraction layers
- Update all extensions to support TabPFN v2
- Improve error reporting and user feedback
- Enhance parameter validation and sensible defaults

### Phase 3: Advanced Features (6-12 months)
- Implement backend-specific optimizations behind common interfaces
- Develop benchmarking framework for extension performance
- Create migration tools for users upgrading between backends
- Refine type system to be more specific and helpful
- Implement runtime compatibility detection and warnings

### Phase 4: Ecosystem Growth (12+ months)
- Develop extension plugin architecture for community contributions
- Create comprehensive benchmarking dashboard
- Implement automatic compatibility testing for PRs
- Build extension registry and discovery mechanism
- Create interactive documentation and examples
