# Guide to TabPFN Extensions Testing Framework

This document explains the testing framework in `conftest.py` that enables testing TabPFN extensions with different TabPFN backends.

## TabPFN Backend Detection

The framework automatically detects which TabPFN implementations are available:

```python
# Global flags indicating what's available
HAS_TABPFN = False         # Is TabPFN package available?
HAS_TABPFN_CLIENT = False  # Is TabPFN client available?
HAS_ANY_TABPFN = False     # Is any implementation available?
TABPFN_SOURCE = None       # Which implementation to use by default
```

These flags are set by trying to import the different implementations:

```python
# Try to import TabPFN package
try:
    import tabpfn
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    HAS_TABPFN = True
    HAS_ANY_TABPFN = True
    TABPFN_SOURCE = "tabpfn"
except ImportError:
    pass

# Try to import TabPFN client
try:
    import tabpfn_client
    from tabpfn_client import TabPFNClassifier as ClientTabPFNClassifier
    from tabpfn_client import TabPFNRegressor as ClientTabPFNRegressor
    HAS_TABPFN_CLIENT = True
    HAS_ANY_TABPFN = True
    if TABPFN_SOURCE is None:
        TABPFN_SOURCE = "tabpfn_client"
except ImportError:
    pass
```

## Test Markers

The framework defines several pytest markers for indicating test compatibility:

```python
def pytest_configure(config):
    config.addinivalue_line("markers", 
                           "requires_tabpfn: mark test to require TabPFN package")
    config.addinivalue_line("markers", 
                           "requires_tabpfn_client: mark test to require TabPFN client")
    config.addinivalue_line("markers", 
                           "requires_any_tabpfn: mark test to require either TabPFN or TabPFN client")
    config.addinivalue_line("markers", 
                           "client_compatible: mark test as compatible with TabPFN client")
```

## TabPFN Fixtures

The framework provides fixtures that automatically use the appropriate TabPFN implementation:

```python
@pytest.fixture
def tabpfn_classifier(request):
    """Return appropriate TabPFN classifier based on availability."""
    if not HAS_ANY_TABPFN:
        raise ValueError("No TabPFN implementation available")
    
    # Check if test is marked to require specific implementation
    requires_tabpfn = request.node.get_closest_marker("requires_tabpfn")
    requires_client = request.node.get_closest_marker("requires_tabpfn_client")
    
    # Return appropriate classifier
    if TABPFN_SOURCE == "tabpfn":
        return TabPFNClassifier(device="cpu")
    elif TABPFN_SOURCE == "tabpfn_client":
        return ClientTabPFNClassifier()
    else:
        pytest.skip("No TabPFN classifier available")
```

A similar fixture exists for `tabpfn_regressor`.

## Helper Functions

The framework also provides helper functions to get all available TabPFN implementations:

```python
def get_all_tabpfn_classifiers():
    """Return TabPFN classifiers from all available sources."""
    classifiers = {}
    
    if HAS_TABPFN:
        from tabpfn import TabPFNClassifier
        classifiers["tabpfn"] = TabPFNClassifier(device="cpu")
    
    if HAS_TABPFN_CLIENT:
        from tabpfn_client import TabPFNClassifier as ClientTabPFNClassifier
        classifiers["tabpfn_client"] = ClientTabPFNClassifier()
    
    return classifiers
```

## Automatic Test Skipping

The framework automatically skips tests based on the available implementations and the test markers:

```python
def pytest_runtest_setup(item):
    """Skip tests based on markers and available implementations."""
    # Skip if test requires TabPFN package but it's not available
    if item.get_closest_marker("requires_tabpfn") and not HAS_TABPFN:
        pytest.skip("Test requires TabPFN package")
    
    # Skip if test requires TabPFN client but it's not available
    if item.get_closest_marker("requires_tabpfn_client") and not HAS_TABPFN_CLIENT:
        pytest.skip("Test requires TabPFN client")
    
    # Skip if test requires any TabPFN but none is available
    if item.get_closest_marker("requires_any_tabpfn") and not HAS_ANY_TABPFN:
        pytest.skip("Test requires either TabPFN or TabPFN client")
    
    # If using TabPFN client, skip tests that are not marked as client compatible
    if TABPFN_SOURCE == "tabpfn_client" and not item.get_closest_marker("client_compatible"):
        pytest.skip("Test is not compatible with TabPFN client")
```

## Using the Framework in Tests

To use this framework in your tests:

1. Add the appropriate markers to your tests:
   ```python
   @pytest.mark.client_compatible  # Works with TabPFN client
   @pytest.mark.requires_any_tabpfn  # Requires any TabPFN implementation
   def test_my_feature(tabpfn_classifier):
       # Your test code here
   ```

2. Use the provided fixtures:
   ```python
   def test_my_feature(tabpfn_classifier):
       # tabpfn_classifier is automatically the appropriate implementation
   ```

3. If you need to test with multiple implementations, use the helper functions:
   ```python
   def test_multiple_implementations():
       classifiers = get_all_tabpfn_classifiers()
       for name, clf in classifiers.items():
           # Test with each available implementation
   ```