from __future__ import annotations

import importlib.util
import os
import sys
from collections.abc import Generator
from pathlib import Path

import pytest


def get_example_files() -> list[dict]:
    """Get all Python files from the examples directory with metadata."""
    package_root = Path(__file__).parent.parent
    examples_dir = package_root / "examples"

    # Examples that require TabPFN core (not client compatible)
    # These should be marked with requires_tabpfn
    tabpfn_only_patterns = ["post_hoc_ensembles/", "unsupervised/", "phe/"]

    # Examples that can work with TabPFN client (client compatible)
    client_compatible_patterns = [
        "classifier_as_regressor/",
        "many_class/",
        "rf_pfn/",
        "interpretability/",
    ]

    # Skip only specific example files that are either tested elsewhere or too slow to run
    # Only skip examples that would be too slow or resource-intensive
    always_skip_patterns = [
        # Large dataset example is too slow to run in regular tests
        "large_datasets_example.py",
    ]

    # Get all example files
    all_file_paths = list(examples_dir.glob("**/*.py"))
    all_files = []

    # Create dictionaries with file path and metadata
    for file_path in all_file_paths:
        rel_path = str(file_path.relative_to(package_root))
        file_name = file_path.name

        file_info = {
            "path": file_path,
            "requires_tabpfn": False,
            "client_compatible": True,
            "name": file_name,
            "always_skip": False,  # By default, don't skip
            "timeout": None,  # No timeout by default - let test execute fully
        }

        # Mark examples to always skip - now we skip all .py files
        if any(pattern in file_name for pattern in always_skip_patterns):
            file_info["always_skip"] = True

        # Mark TabPFN-only examples
        if any(pattern in rel_path for pattern in tabpfn_only_patterns):
            file_info["requires_tabpfn"] = True
            file_info["client_compatible"] = False
        # Mark client-compatible examples
        elif any(pattern in rel_path for pattern in client_compatible_patterns):
            file_info["client_compatible"] = True
        # Default - assume requires TabPFN core
        else:
            file_info["requires_tabpfn"] = True

        all_files.append(file_info)

    return all_files


def import_module_from_path(path: Path, timeout: int | None = None) -> object:
    """Dynamically import a Python module from a file path with timeout.

    Args:
        path: Path to the Python file to import
        timeout: Maximum time to wait for import to complete (in seconds)
               None means no timeout (unlimited time)

    Returns:
        The imported module object

    Raises:
        TimeoutError: If import takes longer than the specified timeout
        ImportError: If the module cannot be imported due to missing dependencies
    """
    import signal
    from contextlib import contextmanager

    @contextmanager
    def time_limit(seconds):
        def signal_handler(signum, frame):
            raise TimeoutError(f"Timed out after {seconds} seconds")

        # Set default timeout of 30 seconds for examples unless specified otherwise
        if seconds is None:
            seconds = 120

        if (
            seconds is not None and sys.platform != "win32"
        ):  # timeout doesn't work on Windows
            signal.signal(signal.SIGALRM, signal_handler)
            signal.alarm(seconds)
            try:
                yield
            finally:
                signal.alarm(0)
        else:
            yield

    # Add the parent directory to sys.path to allow imports within example files
    parent_dir = str(path.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    # Set TEST_MODE environment variable to enable faster execution in the examples
    os.environ["TEST_MODE"] = "1"

    # Use specified timeout (may be None for no timeout)
    with time_limit(timeout):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


@pytest.fixture
def example_files() -> Generator[list[dict], None, None]:
    """Fixture that provides all example files."""
    files = get_example_files()
    if not files:
        pytest.skip("No example files found")
    return files


# Moved pytest_addoption to conftest.py


@pytest.mark.example
@pytest.mark.parametrize("example_file", get_example_files(), ids=lambda x: x["name"])
def test_example(request, example_file):
    """Run example files to ensure they work as expected.

    This test will:
    1. Skip all examples by default unless --run-examples flag is provided
    2. Skip large dataset examples even when --run-examples is provided
    3. Run other examples with appropriate backend compatibility checks

    To run examples: pytest tests/test_examples.py --run-examples
    """
    file_name = example_file["name"]
    example_file["path"]

    # Skip all examples by default, unless --run-examples flag is provided
    if not request.config.getoption("--run-examples"):
        pytest.skip(f"Example {file_name} skipped (use --run-examples to run)")

    # Always skip large dataset examples
    if example_file["always_skip"]:
        pytest.skip(f"Example {file_name} skipped in test environment - too resource intensive")

    # Check if this example requires TabPFN core
    if example_file["requires_tabpfn"]:
        # Get the conftest.py global variable to check if TabPFN core is available
        from conftest import HAS_TABPFN

        if not HAS_TABPFN:
            pytest.fail(
                f"Example {file_name} requires TabPFN core package, but it's not installed",
            )

    # Check if this example requires client compatibility
    if not example_file["client_compatible"]:
        # Using client - verify we're not using TabPFN via client
        from conftest import TABPFN_SOURCE

        if TABPFN_SOURCE == "tabpfn_client":
            pytest.fail(
                f"Example {file_name} is not compatible with TabPFN client, but only client is installed",
            )

    try:
        # Execute the example with the appropriate timeout
        timeout = example_file.get("timeout", None)
        import_module_from_path(example_file["path"], timeout=timeout)
    except TimeoutError as e:
        # Fail on timeouts - don't skip
        pytest.fail(f"Example {file_name} timed out: {e!s}")
    except ImportError as e:
        # If the import error is not related to TabPFN, it's a real issue that should fail
        if "tabpfn" not in str(e).lower():
            pytest.fail(f"Failed to import {file_name}: {e!s}")
        # If it's related to TabPFN, we already checked TabPFN availability above
        # so this is an unexpected TabPFN-related import issue
        pytest.fail(f"Example {file_name} failed to import TabPFN correctly: {e!s}")
    except Exception as e:
        # All exceptions should fail the test - don't skip
        pytest.fail(f"Example {file_name} raised error: {e!s}")


if __name__ == "__main__":
    pytest.main([__file__])
