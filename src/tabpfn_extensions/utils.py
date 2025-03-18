#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0
from __future__ import annotations

import os
import warnings
import numpy as np
from typing import TYPE_CHECKING, Any, Literal, Protocol


class TabPFNEstimator(Protocol):
    def fit(self, X: Any, y: Any) -> Any: ...

    def predict(self, X: Any) -> Any: ...


def is_tabpfn(estimator: Any) -> bool:
    """Check if an estimator is a TabPFN model."""
    try:
        return any(
            [
                "TabPFN" in str(estimator.__class__),
                "TabPFN" in str(estimator.__class__.__bases__),
                any("TabPFN" in str(b) for b in estimator.__class__.__bases__),
                "tabpfn.base_model.TabPFNBaseModel" in str(estimator.__class__.mro()),
            ],
        )
    except (AttributeError, TypeError):
        return False


def get_device(device: str | None = "auto") -> str:
    """Determine the appropriate device for computation.

    This function implements automatic device selection, defaulting to CUDA
    if available, otherwise falling back to CPU.

    Args:
        device: Device specification, options are:
            - "auto": Automatically use CUDA if available, otherwise CPU
            - "cpu": Force CPU usage
            - "cuda": Force CUDA usage (raises error if not available)
            - None: Same as "auto"

    Returns:
        str: The resolved device string ("cpu" or "cuda")

    Raises:
        RuntimeError: If "cuda" is explicitly requested but not available
    """
    import torch

    if device is None or device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device requested but not available. "
            "Use device='auto' to fall back to CPU automatically.",
        )
    return device


USE_TABPFN_LOCAL = os.getenv("USE_TABPFN_LOCAL", "true").lower() == "true"


def get_tabpfn_models() -> tuple[type, type]:
    """Get TabPFN models with fallback between different versions.

    Attempts to import TabPFN models in the following order:
    1. Standard TabPFN package (if USE_TABPFN_LOCAL is True)
    2. TabPFN client

    Returns:
        tuple[type, type]: A tuple containing (TabPFNClassifier, TabPFNRegressor) classes

    Raises:
        ImportError: If none of the TabPFN implementations could be imported
    """

    # First try standard TabPFN package (if local usage is enabled)
    if USE_TABPFN_LOCAL:
        try:
            from tabpfn import TabPFNClassifier, TabPFNRegressor

            if os.getenv("TABPFN_DEBUG", "false").lower() == "true":
                print("Using TabPFN package")

            return TabPFNClassifier, TabPFNRegressor
        except ImportError:
            pass

    # Finally try TabPFN client
    try:
        from tabpfn_client import (
            TabPFNClassifier as ClientTabPFNClassifier,
            TabPFNRegressor as ClientTabPFNRegressor,
        )

        if os.getenv("TABPFN_DEBUG", "false").lower() == "true":
            print("Using TabPFN client")

        # Wrapper classes to add device parameter
        # we can't use *args because scikit-learn needs to know the parameters of the constructor
        class TabPFNClassifierWrapper(ClientTabPFNClassifier):
            def __init__(
                self,
                device: str | None = None,
                categorical_features_indices: list[int] | None = None,
                model_path: str = "default",
                n_estimators: int = 4,
                softmax_temperature: float = 0.9,
                balance_probabilities: bool = False,
                average_before_softmax: bool = False,
                ignore_pretraining_limits: bool = False,
                inference_precision: Literal["autocast", "auto"] = "auto",
                random_state: int
                | np.random.RandomState
                | np.random.Generator
                | None = None,
                inference_config: dict | None = None,
                paper_version: bool = False,
            ) -> None:
                self.device = device
                # Categorical features need to be passed but are not used by client
                self.categorical_features_indices = categorical_features_indices
                if categorical_features_indices is not None:
                    warnings.warn(
                        "categorical_features_indices is not supported in the client version of TabPFN and will be ignored",
                        UserWarning,
                        stacklevel=2,
                    )
                if "/" in model_path:
                    model_name = model_path.split("/")[-1].split("-")[-1].split(".")[0]
                    if model_name == "classifier":
                        model_name = "default"
                    self.model_path = model_name
                else:
                    self.model_path = model_path

                super().__init__(
                    model_path=self.model_path,
                    n_estimators=n_estimators,
                    softmax_temperature=softmax_temperature,
                    balance_probabilities=balance_probabilities,
                    average_before_softmax=average_before_softmax,
                    ignore_pretraining_limits=ignore_pretraining_limits,
                    inference_precision=inference_precision,
                    random_state=random_state,
                    inference_config=inference_config,
                    paper_version=paper_version,
                )

            def get_params(self, deep: bool = True) -> dict[str, Any]:
                """Return parameters for this estimator."""
                params = super().get_params(deep=deep)
                params.pop("device")
                params.pop("categorical_features_indices")
                return params

        class TabPFNRegressorWrapper(ClientTabPFNRegressor):
            def __init__(
                self,
                device: str | None = None,
                categorical_features_indices: list[int] | None = None,
                model_path: str = "default",
                n_estimators: int = 8,
                softmax_temperature: float = 0.9,
                average_before_softmax: bool = False,
                ignore_pretraining_limits: bool = False,
                inference_precision: Literal["autocast", "auto"] = "auto",
                random_state: int
                | np.random.RandomState
                | np.random.Generator
                | None = None,
                inference_config: dict | None = None,
                paper_version: bool = False,
            ) -> None:
                self.device = device
                self.categorical_features_indices = categorical_features_indices
                if categorical_features_indices is not None:
                    warnings.warn(
                        "categorical_features_indices is not supported in the client version of TabPFN and will be ignored",
                        UserWarning,
                        stacklevel=2,
                    )

                if "/" in model_path:
                    model_name = model_path.split("/")[-1].split("-")[-1].split(".")[0]
                    if model_name == "regressor":
                        model_name = "default"
                    self.model_path = model_name
                else:
                    self.model_path = model_path

                super().__init__(
                    model_path=self.model_path,
                    n_estimators=n_estimators,
                    softmax_temperature=softmax_temperature,
                    average_before_softmax=average_before_softmax,
                    ignore_pretraining_limits=ignore_pretraining_limits,
                    inference_precision=inference_precision,
                    random_state=random_state,
                    inference_config=inference_config,
                    paper_version=paper_version,
                )

            def get_params(self, deep: bool = True) -> dict[str, Any]:
                """Return parameters for this estimator."""
                params = super().get_params(deep=deep)
                params.pop("device")
                params.pop("categorical_features_indices")
                return params

        return TabPFNClassifierWrapper, TabPFNRegressorWrapper

    except ImportError:
        raise ImportError(
            "No TabPFN implementation could be imported. Install with one of the following:\n"
            "pip install tabpfn    # For standard TabPFN package\n"
            "pip install tabpfn-client  # For TabPFN client (API-based inference)",
        )


TabPFNClassifier, TabPFNRegressor = get_tabpfn_models()


def infer_categorical_features(
    X: np.ndarray, categorical_features: list[int] | None = None
) -> list[int]:
    """Infer the categorical features from the input data.
    
    We take `categorical_features` as the initial list of categorical features
    and refine it based on the number of unique values in each feature.

    Parameters:
        X (np.ndarray): The input data.
        categorical_features (list[int], optional): Initial list of categorical feature indices.
            If None, will start with an empty list.

    Returns:
        list[int]: The indices of the categorical features.
    """
    if categorical_features is None:
        categorical_features = []
        
    max_unique_values_as_categorical_feature = 10
    min_unique_values_as_numerical_feature = 10

    _categorical_features: list[int] = []
    for i in range(X.shape[-1]):
        # Filter categorical features, with too many unique values
        if i in categorical_features and (
            len(np.unique(X[:, i])) <= max_unique_values_as_categorical_feature
        ):
            _categorical_features += [i]

        # Filter non-categorical features, with few unique values
        elif (
            i not in categorical_features
            and len(np.unique(X[:, i])) < min_unique_values_as_numerical_feature
            and X.shape[0] > 100
        ):
            _categorical_features += [i]

    return _categorical_features
