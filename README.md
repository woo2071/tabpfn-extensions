# 🎯 TabPFN Extensions

⚡ Build powerful extensions for the world's most efficient tabular foundation model ⚡

[![PyPI version](https://badge.fury.io/py/tabpfn.svg)](https://badge.fury.io/py/tabpfn)
[![Downloads](https://pepy.tech/badge/tabpfn)](https://pepy.tech/project/tabpfn)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Discord](https://img.shields.io/discord/1234567890?color=7289da&label=Discord&logo=discord&logoColor=ffffff)](https://discord.com/channels/1285598202732482621/)
[![Twitter Follow](https://img.shields.io/twitter/follow/PriorLabs?style=social)](https://twitter.com/PriorLabs)

Looking for the main TabPFN package? Check out [TabPFN](https://github.com/priorlabs/tabpfn).

## Quick Install

```bash
pip install -e .
```

Depending on if you have a GPU available and would like to run locally or via web client use
pip install tabpfn
or pip install tabpfn-client

## 🤔 What is TabPFN Extensions?

**TabPFN Extensions** is a collection of community-driven extensions and tools built around TabPFN, the state-of-the-art
foundation model for tabular data. This repository makes it easy to:

- Build domain-specific extensions
- Create integrations with other frameworks
- Share utilities and tools
- Contribute example applications
- Develop custom solutions

## 🚀 Featured Extensions

Here are some highlighted community extensions:

**🔮 Unsupervised Learning**

- Data generation capabilities
- Outlier detection
- Distribution modeling

**🔍 Interpretability**

- Feature importance analysis
- Model explanation tools
- Decision boundary visualization

**⚡ AutoTabPFN**

- Post-hoc ensemble techniques
- Automatic hyperparameter tuning
- Optimized performance

**🌲 Random Forest PFN**

- Random forest adaptation of TabPFN
- Scalable for larger datasets
- Parallel processing support

And many more! Browse the [full list of extensions](https://github.com/priorlabs/tabpfn_extensions/tree/main/src).

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

## 📖 Documentation

- [Getting Started](https://docs.priorlabs.ai/getting-started)
- [API Reference](https://docs.priorlabs.ai/api)
- [Examples](https://docs.priorlabs.ai/examples)
- [Contributing Guide](CONTRIBUTING.md)

## 🌟 Community

- Join our [Discord](https://discord.gg/tabpfn)
- Follow us on [Twitter](https://twitter.com/PriorLabs)
- Read our [Blog](https://blog.priorlabs.ai)

## 🤝 License

This project is licensed under the Apache License 2.0 - see the [LICENSE.txt](LICENSE.txt) file for details.

## 🌟 Contributors

[![tabpfn-extensions contributors](https://contrib.rocks/image?repo=priorlabs/tabpfn-extensions&max=2000)](https://github.com/priorlabs/tabpfn-extensions/graphs/contributors)

## 📚 Citation

```bibtex
@article{hollmann2024tabpfn,
  title={Accurate predictions on small data with a tabular foundation model},
  author={Hollmann, Noah and M{\"u}ller, Samuel and Purucker, Lennart and 
          Krishnakumar, Arjun and K{\"o}rfer, Max and Hoo, Shi Bin and 
          Schirrmeister, Robin Tibor and Hutter, Frank},
  journal={Nature},
  year={2024},
  month={01},
  day={09},
  doi={10.1038/s41586-024-08328-6},
  publisher={Springer Nature},
  url={https://doi.org/10.1038/s41586-024-08328-6},
}
```
---

Built with ❤️ by the TabPFN community