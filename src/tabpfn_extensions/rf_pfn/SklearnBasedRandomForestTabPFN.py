"""Backward compatibility wrapper for RandomForestTabPFN.

This module simply imports from the snake_case version to maintain backward compatibility.
Use sklearn_based_random_forest_tabpfn.py directly for new code.
"""

#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

import warnings

from .sklearn_based_random_forest_tabpfn import (
    RandomForestTabPFNBase,
    RandomForestTabPFNClassifier,
    RandomForestTabPFNRegressor,
    _accumulate_prediction,
    softmax_numpy,
)

warnings.warn(
    "SklearnBasedRandomForestTabPFN.py is deprecated and will be removed in a future version. "
    "Please use 'from tabpfn_extensions.rf_pfn import RandomForestTabPFNClassifier, "
    "RandomForestTabPFNRegressor' instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export for backward compatibility
__all__ = [
    "RandomForestTabPFNBase",
    "RandomForestTabPFNClassifier",
    "RandomForestTabPFNRegressor",
    "_accumulate_prediction",
    "softmax_numpy",
]