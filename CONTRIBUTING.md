# Contributing to TabPFN Extensions

Welcome to TabPFN Extensions! This repository is a collection of community-contributed packages that extend and enhance TabPFN, a foundation model for tabular data. We welcome all contributions and aim to make the process as simple as possible.

## 📋 Quick Start

1. **Setup your development environment**:

```bash
# Clone the repository
git clone https://github.com/PriorLabs/tabpfn-extensions.git
cd tabpfn-extensions

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install for development with all dependencies
pip install -e ".[dev,all]"

# Install pre-commit hooks
pre-commit install
```

2. **Create your extension package**:

```bash
mkdir -p src/tabpfn_extensions/your_package examples/your_package tests/
```

3. **Start coding**:
   - Implement your extension in `src/tabpfn_extensions/your_package/`
   - Create examples in `examples/your_package/`
   - Write tests in `tests/test_your_package.py`

4. **Test your code**:

```bash
# Run all tests
pytest

# Run tests for your package
pytest tests/test_your_package.py
```

## 🔄 TabPFN Compatibility

TabPFN Extensions supports two different TabPFN implementations:

1. **TabPFN Package** - Full PyTorch implementation for local inference
2. **TabPFN Client** - Lightweight API client for cloud-based inference

To ensure your extension works with both backends:

1. **Import TabPFN in a flexible way**:

```python
try:
    # Try standard TabPFN package first
    from tabpfn import TabPFNClassifier, TabPFNRegressor
except ImportError:
    # Fall back to TabPFN client
    from tabpfn_client import TabPFNClassifier, TabPFNRegressor
```

2. **Use common parameters** that are available in both implementations.

3. **Add appropriate test markers** in your tests:
   - `client_compatible` if your extension works with TabPFN client
   - `requires_tabpfn` if it requires the full TabPFN package

## 📁 Repository Structure

TabPFN Extensions uses a modular structure where each contribution lives in its own subpackage:

```
tabpfn-extensions/
├── src/
│   └── tabpfn_extensions/
│       └── your_package/       # Your extension code
├── examples/
│   └── your_package/           # Usage examples
└── tests/
    └── test_your_package.py    # Tests for your extension
```

## 📝 Code Guidelines

### Requirements

- Python 3.9+ compatibility
- Support for both TabPFN and TabPFN-client when possible
- Minimize dependencies and document them in pyproject.toml
- Follow scikit-learn API conventions when appropriate
- Document all public functions and classes

### Documentation

Each extension should include:

- Clear docstrings using Google style format
- Type hints for all function parameters and return values
- At least one example script in the examples directory
- Comments explaining non-obvious code sections

### Testing

- Write tests using pytest
- Use appropriate markers: `client_compatible`, `requires_tabpfn`, `requires_any_tabpfn`
- Test both TabPFN and TabPFN-client compatibility where appropriate
- Use the fixtures provided in conftest.py for TabPFN instances

## ✅ Contribution Checklist

- [ ] Added extension under src/tabpfn_extensions/
- [ ] Included example(s) in examples/
- [ ] Added appropriate dependencies to pyproject.toml
- [ ] Written tests in tests/
- [ ] Documented all public functions and classes
- [ ] Ensured compatibility with both TabPFN backends (when possible)
- [ ] Run all linting and type checking tools

## 📜 Legal & Security

- Only contribute code you have rights to
- Don't include sensitive or private data
- All contributions must be Apache 2.0 licensed
- No machine learning models or large datasets in the repository

## 📦 Example Extension Structure

Here's a minimal example of a complete extension:

```
src/tabpfn_extensions/sample_extension/
├── __init__.py
├── core.py
└── utils.py

examples/sample_extension/
├── basic_usage.py
└── advanced_features.py

tests/
└── test_sample_extension.py
```

## 🧪 Versioning and Releases

We follow [Semantic Versioning](https://semver.org/), with version numbers in the format MAJOR.MINOR.PATCH:

- MAJOR: Incompatible API changes
- MINOR: New functionality in a backwards compatible manner
- PATCH: Backwards compatible bug fixes

## 🛠️ Need Help?

- Join our [Discord](https://discord.com/channels/1285598202732482621/) for questions
- Open an issue for bug reports
- Check out existing extensions for examples
