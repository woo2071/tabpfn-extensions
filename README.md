# TabPFN Extensions ⚡

[![PyPI version](https://badge.fury.io/py/tabpfn.svg)](https://badge.fury.io/py/tabpfn)
[![Downloads](https://pepy.tech/badge/tabpfn)](https://pepy.tech/project/tabpfn)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Discord](https://img.shields.io/discord/1285598202732482621?color=7289da&label=Discord&logo=discord&logoColor=ffffff)](https://discord.com/channels/1285598202732482621/)
[![Twitter Follow](https://img.shields.io/twitter/follow/Prior_Labs?style=social)](https://twitter.com/Prior_Labs)
![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen)
![Last Commit](https://img.shields.io/github/last-commit/automl/tabpfn-client)

<img src="tabpfn_summary.webp" width="650" alt="TabPFN Summary">

## 📋 Overview

TabPFN Extensions is a collection of utilities and extensions for [TabPFN](https://github.com/priorlabs/tabpfn), a foundation model for tabular data. These extensions enhance TabPFN's capabilities with:

- 🧪 **Post-hoc ensembles**: Combine multiple TabPFN models for better performance
- 🔍 **Interpretability**: Understand model decisions with SHAP and feature selection
- 🎯 **Hyperparameter optimization**: Automatically tune TabPFN for best results
- 🔄 **Adaptation tools**: Convert classifiers to regressors and handle multi-class problems
- 🌳 **Tree-based ensembles**: Combine TabPFN with decision trees and random forests

## ⚙️ Installation

```bash
# Basic installation with core dependencies
pip install tabpfn-extensions

# Clone and install the repository (alternative method)
pip install "tabpfn-extensions[post_hoc_ensembles,interpretability,hpo] @ git+https://github.com/PriorLabs/tabpfn-extensions.git"
# or choose below method if you need to have source code
# git clone https://github.com/PriorLabs/tabpfn-extensions
# pip install -e "tabpfn-extensions[post_hoc_ensembles,interpretability,hpo]"

# Install specific extension modules
pip install "tabpfn-extensions[interpretability,hpo]"

# Install all extensions
pip install "tabpfn-extensions[all]"

# Development installation (minimal)
pip install -e ".[dev]"

# Development installation (all extensions - slower)
pip install -e ".[dev,all]"
```

### 🔄 Backend Options

TabPFN Extensions works with two TabPFN implementations:

1. **🖥️ TabPFN Package** - Full PyTorch implementation for local inference:
   ```bash
   pip install tabpfn
   ```

2. **☁️ TabPFN Client** - Lightweight API client for cloud-based inference:
   ```bash
   pip install tabpfn-client
   ```

Choose the backend that fits your needs - most extensions work with either option!

## 🚀 Quick Start

```python
from tabpfn_extensions.many_class import ManyClassClassifier
from tabpfn import TabPFNClassifier  # or from tabpfn_client import TabPFNClassifier

# Create a base TabPFN classifier
base_clf = TabPFNClassifier()

# Wrap it to handle more than 10 classes
many_class_clf = ManyClassClassifier(estimator=base_clf, alphabet_size=10)

# Fit and predict as usual
many_class_clf.fit(X_train, y_train)
y_pred = many_class_clf.predict(X_test)
```

## 🧑‍💻 For Contributors

Interested in adding your own extension? We welcome contributions!

```bash
# Clone and set up for development
git clone https://github.com/PriorLabs/tabpfn-extensions.git
cd tabpfn-extensions

# Lightweight dev setup (fast)
pip install -e ".[dev]"

# Test your extension with fast mode
FAST_TEST_MODE=1 pytest tests/test_your_extension.py -v
```

See our [Contribution Guide](CONTRIBUTING.md) for more details.

## 🛠️ Available Extensions

- **post_hoc_ensembles**: Improve performance with model combination
- **interpretability**: Explain TabPFN predictions with SHAP values and feature selection
- **many_class**: Handle classification with more classes than TabPFN's default limit
- **classifier_as_regressor**: Use TabPFN's classifier for regression tasks
- **hpo**: Automatic hyperparameter tuning for TabPFN
- **rf_pfn**: Combine TabPFN with decision trees and random forests
- **unsupervised**: Data generation and outlier detection

Detailed documentation for each extension is available in the respective module directories.

## 📦 Repository Structure

Each extension lives in its own subpackage:

```
tabpfn-extensions/
├── src/
│   └── tabpfn_extensions/
│       └── your_package/      # Extension implementation
├── examples/
│   └── your_package/          # Usage examples
└── tests/
    └── your_package/          # Tests
```

## 🤝 Contributing

We welcome all contributions! See our [Contributing Guide](CONTRIBUTING.md) for details.

Quick start:

1. Fork the repository
2. Create your package under `src/`
3. Ensure compatibility with both TabPFN implementations
4. Add examples and tests
5. Submit a PR

## 📝 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

Built with ❤️ by the TabPFN community
