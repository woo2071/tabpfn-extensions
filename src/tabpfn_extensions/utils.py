#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

import os

import os
from typing import Any, Type, Tuple, Protocol, Literal
from dataclasses import dataclass
from typing_extensions import override

class TabPFNEstimator(Protocol):
    def fit(self, X: Any, y: Any) -> Any:
        ...

    def predict(self, X: Any) -> Any:
        ...


def is_tabpfn(estimator: Any) -> bool:
    """Check if an estimator is a TabPFN model."""
    try:
        return any(
            [
                "TabPFN" in str(estimator.__class__),
                "TabPFN" in str(estimator.__class__.__bases__),
                any("TabPFN" in str(b) for b in estimator.__class__.__bases__),
                "tabpfn.base_model.TabPFNBaseModel" in str(estimator.__class__.mro()),
            ]
        )
    except (AttributeError, TypeError):
        return False
    

@dataclass
class PreprocessorConfigDefault:
    """Configuration for data preprocessors.

    Attributes:
        name: Name of the preprocessor.
        categorical_name:
            Name of the categorical encoding method.
            Options: "none", "numeric", "onehot", "ordinal", "ordinal_shuffled", "none".
        append_original: Whether to append original features to the transformed features
        subsample_features: Fraction of features to subsample. -1 means no subsampling.
        global_transformer_name: Name of the global transformer to use.
    """

    name: Literal[
        "per_feature",  # a different transformation for each feature
        "power",  # a standard sklearn power transformer
        "safepower",  # a power transformer that prevents some numerical issues
        "power_box",
        "safepower_box",
        "quantile_uni_coarse",  # quantile transformations with few quantiles up to many
        "quantile_norm_coarse",
        "quantile_uni",
        "quantile_norm",
        "quantile_uni_fine",
        "quantile_norm_fine",
        "robust",  # a standard sklearn robust scaler
        "kdi",
        "none",  # no transformation (only standardization in transformer)
        "kdi_random_alpha",
        "kdi_uni",
        "kdi_random_alpha_uni",
        "adaptive",
        "norm_and_kdi",
        # KDI with alpha collection
        "kdi_alpha_0.3_uni",
        "kdi_alpha_0.5_uni",
        "kdi_alpha_0.8_uni",
        "kdi_alpha_1.0_uni",
        "kdi_alpha_1.2_uni",
        "kdi_alpha_1.5_uni",
        "kdi_alpha_2.0_uni",
        "kdi_alpha_3.0_uni",
        "kdi_alpha_5.0_uni",
        "kdi_alpha_0.3",
        "kdi_alpha_0.5",
        "kdi_alpha_0.8",
        "kdi_alpha_1.0",
        "kdi_alpha_1.2",
        "kdi_alpha_1.5",
        "kdi_alpha_2.0",
        "kdi_alpha_3.0",
        "kdi_alpha_5.0",
    ]
    categorical_name: Literal[
        # categorical features are pretty much treated as ordinal, just not resorted
        "none",
        # categorical features are treated as numeric,
        # that means they are also power transformed for example
        "numeric",
        # "onehot": categorical features are onehot encoded
        "onehot",
        # "ordinal": categorical features are sorted and encoded as
        # integers from 0 to n_categories - 1
        "ordinal",
        # "ordinal_shuffled": categorical features are encoded as integers
        # from 0 to n_categories - 1 in a random order
        "ordinal_shuffled",
        "ordinal_very_common_categories_shuffled",
    ] = "none"
    append_original: bool = False
    subsample_features: float = -1
    global_transformer_name: str | None = None

    @override
    def __str__(self) -> str:
        return (
            f"{self.name}_cat:{self.categorical_name}"
            + ("_and_none" if self.append_original else "")
            + (
                f"_subsample_feats_{self.subsample_features}"
                if self.subsample_features > 0
                else ""
            )
            + (
                f"_global_transformer_{self.global_transformer_name}"
                if self.global_transformer_name is not None
                else ""
            )
        )


from typing import Tuple, Type
import os
USE_TABPFN_LOCAL = os.getenv("USE_TABPFN_LOCAL", "true").lower() == "true"

def get_tabpfn_models() -> Tuple[Type, Type, Type]:
    """Get TabPFN models with fallback between local and client versions."""
    if USE_TABPFN_LOCAL:
        try:
            from tabpfn import TabPFNClassifier, TabPFNRegressor
            from tabpfn.preprocessing import PreprocessorConfig

            return TabPFNClassifier, TabPFNRegressor, PreprocessorConfig
        except ImportError:
            pass

    try:
        from tabpfn_client import (
            TabPFNClassifier as ClientTabPFNClassifier,
            TabPFNRegressor as ClientTabPFNRegressor,
        )

        # Wrapper classes to add device parameter
        class TabPFNClassifier(ClientTabPFNClassifier):
            def __init__(self, *args, device=None, **kwargs):
                super().__init__(*args, **kwargs)
                # Ignoring the device parameter for now

        class TabPFNRegressor(ClientTabPFNRegressor):
            def __init__(self, *args, device=None, **kwargs):
                super().__init__(*args, **kwargs)
                # Ignoring the device parameter for now

        return TabPFNClassifier, TabPFNRegressor, PreprocessorConfigDefault

    except ImportError:
        raise ImportError(
            "Neither local TabPFN nor TabPFN client could be imported. Install with:\n"
            "pip install tabpfn\n"
            "or\n"
            "pip install tabpfn-client"
        )


TabPFNClassifier, TabPFNRegressor, PreprocessorConfig = get_tabpfn_models()
