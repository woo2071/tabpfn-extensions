from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("tabpfn-extensions")
except PackageNotFoundError:
    __version__ = "0.1.0.dev0"

# Create alias for test_utils
from . import test_utils

# Import utilities and wrapped TabPFN classes
from .utils import get_tabpfn_models, is_tabpfn

# Get the TabPFN models with our wrappers applied
TabPFNClassifier, TabPFNRegressor = get_tabpfn_models()

__all__ = ["test_utils", "TabPFNClassifier", "TabPFNRegressor", "is_tabpfn"]
