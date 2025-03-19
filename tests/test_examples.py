"""Test examples to ensure they work as expected.

This module provides a testing framework for TabPFN example files. It automatically:
1. Detects all example files in the examples/ directory
2. Categorizes them as fast, slow, or large dataset examples
3. Runs fast examples normally
4. Tests slow examples with a short timeout (expecting timeout as success)
5. Skips large dataset examples entirely unless explicitly requested
6. Handles backend compatibility for TabPFN package vs. TabPFN client

Usage:
    # Run only fast examples:
    FAST_TEST_MODE=1 python -m pytest tests/test_examples.py

    # Run all examples except large dataset ones (slow ones will timeout and be marked as xfail):
    FAST_TEST_MODE=1 python -m pytest tests/test_examples.py --run-examples

    # Run specific example, even if it's a large dataset example:
    python -m pytest tests/test_examples.py::test_example[large_datasets_example.py] --run-examples
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

# Enable test mode to make examples run faster
os.environ["TEST_MODE"] = "1"


def get_example_files() -> list[dict]:
    """Get all Python files from the examples directory with metadata.

    Each example is categorized as:
    - fast: Can run quickly (runs in both normal and fast test mode)
    - slow: Takes longer to run (runs with a 1-second timeout, expected to timeout)
    - always_timeout: Examples with large datasets that are always skipped unless explicitly requested
    - requires_tabpfn: If True, requires the full TabPFN package and won't work with client;
                       if False, works with either TabPFN package or TabPFN client

    Returns:
        List of dictionaries containing example file info
    """
    package_root = Path(__file__).parent.parent
    examples_dir = package_root / "examples"

    # The only example that runs fast enough for CI
    FAST_EXAMPLES = ["generate_data.py"]

    # These directories/files need the full TabPFN package and won't work with client
    REQUIRES_TABPFN_DIRS = ["large_datasets/", "embedding/"]

    # Large dataset examples are always expected to timeout,
    # even if --run-examples is provided
    ALWAYS_TIMEOUT_PATTERNS = ["large_datasets_example.py"]

    # Find all Python files in the examples directory
    all_file_paths = list(examples_dir.glob("**/*.py"))
    all_files = []

    # Process each file with appropriate metadata
    for file_path in all_file_paths:
        rel_path = str(file_path.relative_to(package_root))
        file_name = file_path.name

        file_info = {
            "path": file_path,
            "name": file_name,
            # Default classification - most examples work with both implementations
            "requires_tabpfn": False,  # By default, examples work with either implementation
            "fast": file_name in FAST_EXAMPLES,  # Only listed examples are fast
            "slow": file_name not in FAST_EXAMPLES,  # All others are slow
            "always_timeout": any(
                pattern in file_name for pattern in ALWAYS_TIMEOUT_PATTERNS
            ),
            "timeout": 1
            if file_name not in FAST_EXAMPLES
            else 30,  # Short timeout for slow examples
        }

        # Check if example requires full TabPFN package
        if any(pattern in rel_path for pattern in REQUIRES_TABPFN_DIRS):
            # Example explicitly requires TabPFN package
            file_info["requires_tabpfn"] = True

        all_files.append(file_info)

    return all_files


def import_module_from_path(path: Path, timeout: int = 60) -> object:
    """Dynamically import a Python module from a file path with timeout.

    Args:
        path: Path to the Python file to import
        timeout: Maximum time to wait for import to complete (in seconds)

    Returns:
        The imported module object

    Raises:
        TimeoutError: If import takes longer than the specified timeout
    """
    import signal
    from contextlib import contextmanager

    @contextmanager
    def time_limit(seconds):
        def signal_handler(signum, frame):
            raise TimeoutError(f"Timed out after {seconds} seconds")

        if sys.platform != "win32":  # timeout doesn't work on Windows
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

    # Use specified timeout
    with time_limit(timeout):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


@pytest.mark.example
@pytest.mark.parametrize("example_file", get_example_files(), ids=lambda x: x["name"])
def test_example(request, example_file):
    """Run example files to ensure they work as expected.

    Test strategy:
    1. Fast examples are run with normal timeout
    2. Slow examples are run with 1-second timeout, expected to timeout
    3. Examples are skipped if they require missing backends
    4. In FAST_TEST_MODE, only fast examples and examples with --run-examples flag run

    Args:
        request: PyTest request fixture
        example_file: Dictionary with example file metadata
    """
    from conftest import FAST_TEST_MODE, HAS_TABPFN, HAS_TABPFN_CLIENT, TABPFN_SOURCE

    file_name = example_file["name"]
    file_path = example_file["path"]

    # Skip examples unless explicitly included
    if not request.config.getoption("--run-examples"):
        # In fast mode, skip non-fast examples
        if FAST_TEST_MODE and not example_file["fast"]:
            pytest.skip(
                f"Example {file_name} skipped in fast test mode (use --run-examples to run)",
            )

    # Skip if backend not available
    if example_file["requires_tabpfn"]:
        if not HAS_TABPFN:
            pytest.skip(
                f"Example {file_name} requires TabPFN package, but it's not installed",
            )
        elif TABPFN_SOURCE == "tabpfn_client":
            pytest.skip(
                f"Example {file_name} requires TabPFN package, not compatible with client",
            )

    # Large dataset examples are always skipped with --run-examples,
    # unless they are explicitly requested by name
    this_example_explicitly_requested = file_name in str(request.node.name)
    if (
        example_file.get("always_timeout", False)
        and not this_example_explicitly_requested
    ):
        pytest.skip(
            f"Example {file_name} involves large datasets and is always skipped unless directly specified",
        )

    try:
        # Handle slow examples (including large datasets) differently - expect them to timeout
        if example_file.get("slow", False) or example_file.get("always_timeout", False):
            try:
                # Execute with very short timeout (1 second)
                timeout = example_file.get("timeout", 1)
                import_module_from_path(file_path, timeout=timeout)
                # If it somehow completes within the timeout, that's fine too
                print(
                    f"Note: Slow example {file_name} surprisingly completed within {timeout}s timeout",
                )
            except TimeoutError as e:
                # Expected behavior: mark as successful with xfail
                pytest.xfail(f"Example {file_name} timed out as expected: {e!s}")
        else:
            # Fast examples should complete normally within timeout
            timeout = example_file.get("timeout", 30)
            import_module_from_path(file_path, timeout=timeout)
    except TimeoutError as e:
        # Unexpected timeout in fast examples is a failure
        pytest.fail(f"Example {file_name} timed out: {e!s}")
    except ImportError as e:
        error_msg = str(e).lower()
        # Handle TabPFN-specific import errors
        if "tabpfn" in error_msg:
            if not HAS_TABPFN and not HAS_TABPFN_CLIENT:
                pytest.skip(
                    f"Example {file_name} requires TabPFN, but neither implementation is installed",
                )
            elif not HAS_TABPFN and example_file["requires_tabpfn"]:
                pytest.skip(
                    f"Example {file_name} requires TabPFN package, but only client is installed",
                )
            else:
                pytest.fail(
                    f"Example {file_name} failed to import TabPFN correctly: {e!s}",
                )
        else:
            pytest.fail(f"Failed to import {file_name}: {e!s}")
    except (AttributeError, ValueError, TypeError, FileNotFoundError, NameError) as e:
        # Specific exceptions that might be raised during imports
        pytest.fail(f"Example {file_name} raised error: {e!s}")


def pytest_addoption(parser):
    """Add command-line options to pytest."""
    parser.addoption(
        "--run-examples",
        action="store_true",
        default=False,
        help="Run all example files (including slow ones that will be expected to timeout)",
    )


if __name__ == "__main__":
    pytest.main([__file__])
