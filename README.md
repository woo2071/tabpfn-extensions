# TabPFN Community Contributions

[![PyPI version](https://badge.fury.io/py/tabpfn.svg)](https://badge.fury.io/py/tabpfn)
[![Downloads](https://pepy.tech/badge/tabpfn)](https://pepy.tech/project/tabpfn)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Discord](https://img.shields.io/discord/1285598202732482621?color=7289da&label=Discord&logo=discord&logoColor=ffffff)](https://discord.com/channels/1285598202732482621/)
[![Twitter Follow](https://img.shields.io/twitter/follow/Prior_Labs?style=social)](https://twitter.com/Prior_Labs)
![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen)
![Last Commit](https://img.shields.io/github/last-commit/automl/tabpfn-client)

<img src="tabpfn_summary.webp" width="650" alt="TabPFN Summary">

## Quick Install

```bash
# Clone and install the repository
pip install "tabpfn-community[post_hoc_ensembles,interpretability,hpo] @ git+https://github.com/PriorLabs/tabpfn-community.git"
# or choose below method if you need to have source code
# git clone https://github.com/PriorLabs/tabpfn-community
# pip install -e "tabpfn-community[post_hoc_ensembles,interpretability,hpo]"

# Choose one of the following installation options:

# 1. For GPU-accelerated local inference:
pip install tabpfn

# 2. For cloud-based inference via API:
pip install tabpfn-client
```

## 🌐 TabPFN Ecosystem

Choose the right TabPFN implementation for your needs:

- **[TabPFN Client](https://github.com/automl/tabpfn-client)**: Easy-to-use API client for cloud-based inference
- **TabPFN Community Contributions (this repo)**: Community extensions and integrations
- **[TabPFN](https://github.com/priorlabs/tabpfn)**: Core implementation for local deployment and research
Browse the [full list of extensions](https://github.com/priorlabs/tabpfn-extensions/tree/main/src/tabpfn_extensions).


## 📦 Repository Structure

Each extension lives in its own subpackage:

```
tabpfn-extensions/
├── src/
│   └── tabpfn_extensions/  
│       └── your_package/      # Your extension code
├── examples/
│   └── your_package/          # Usage examples
├── tests/
│   └── your_package/          # Tests
└── requirements/
    └── your_package.txt       # Dependencies (optional)
```

## 🛠️ Contributing

We welcome all contributions! See our [Contributing Guide](CONTRIBUTING.md) for details.

Quick start:

1. Fork the repository
2. Create your package under `src/`
3. Add examples
4. Submit a PR

## 🤝 License

This project is licensed under the Apache License 2.0 - see the [LICENSE.txt](LICENSE.txt) file for details.

---

Built with ❤️ by the TabPFN community
