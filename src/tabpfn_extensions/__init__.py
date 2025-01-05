from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("your_package")
except PackageNotFoundError:
    __version__ = "0.1.0.dev0"

from . import utils
from . import utils_todo