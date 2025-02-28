#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

"""
This module is deprecated. Use sklearn_based_decision_tree_tabpfn instead.

Maintained for backward compatibility.
"""

from __future__ import annotations

# Re-export the classes from the new module
from .sklearn_based_decision_tree_tabpfn import (
    DecisionTreeTabPFNBase,
    DecisionTreeTabPFNClassifier, 
    DecisionTreeTabPFNRegressor
)

__all__ = ["DecisionTreeTabPFNBase", "DecisionTreeTabPFNClassifier", "DecisionTreeTabPFNRegressor"]