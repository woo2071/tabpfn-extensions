# Contributing

Welcome to TabPFN Systems! This repository is a collection of community-contributed packages that extend and enhance
TabPFN. We welcome all contributions and aim to make contributing as simple as possible.

## Quick Start

1. Install for development:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .
```

2. Create your package:

```bash
mkdir src/your_package examples/your_package
```

3. Start coding!

## Repository Structure

TabPFN Systems uses a modular structure where each contribution lives in its own subpackage:

```
tabpfn_system/
├── src/
│   └── your_package/          # Your package code
├── examples/
│   └── your_package/          # Usage examples
├── tests/
│   └── your_package/          # Tests (optional)
└── requirements/
    └── your_package.txt       # Additional dependencies (if needed)
```

## Guidelines

### Package Requirements

- Python 3.8+
- TabPFN 1.0+
- Keep dependencies minimal and document them
- Avoid large data files (>10MB)
- Include license headers in source files

### Documentation

Each package should include:

- README.md with:
    - Package description
    - Basic usage example
    - Requirements
    - Author(s)
- Docstrings for public functions
- At least one example notebook/script

### Naming & Structure

- Follow standard Python package structure
- One package per contribution

## Checklist

- [ ] Added package under src/
- [ ] Included example(s)
- [ ] Added requirements.txt (if needed)
- [ ] Created README.md

## Legal & Security

- Only contribute code you have rights to
- Don't include sensitive or private data
- All contributions must be Apache 2.0 licensed

## Example Package Structure

Here's a minimal example:

```

src/interpretability/
├── __init__.py
├── shap.py
└── utils.py

examples/interpretability/
├── basic_usage.ipynb
└── README.md

requirements/
└── interpretability.txt
