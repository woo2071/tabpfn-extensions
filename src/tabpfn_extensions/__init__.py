from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("tabpfn-extensions")
except PackageNotFoundError:
    __version__ = "0.1.0.dev0"

from .utils import TabPFNClassifier, TabPFNRegressor, is_tabpfn

# Create alias for test_utils
from . import test_utils
