from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("your_package")
except PackageNotFoundError:
    __version__ = "0.1.0.dev0"

from . import utils_todo
from .utils import TabPFNClassifier, TabPFNRegressor, is_tabpfn
