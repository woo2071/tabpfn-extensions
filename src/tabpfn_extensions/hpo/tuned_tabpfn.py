#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

"""Hyperparameter Optimization (HPO) for TabPFN models.

This module provides automatic tuning capabilities for TabPFN models using
Bayesian optimization via Hyperopt. It finds optimal hyperparameters for both
the TabPFN model and its inference configuration.

Key features:
- Optimized search spaces for classification and regression tasks
- Support for multiple evaluation metrics (accuracy, ROC-AUC, F1, RMSE, MAE)
- Proper handling of categorical features through automatic encoding
- Compatible with both TabPFN and TabPFN-client backends
- Implements scikit-learn's estimator interface for easy integration
- Built-in validation strategies for reliable performance estimation

Example usage:
    ```python
    from tabpfn_extensions.hpo import TunedTabPFNClassifier

    # Create a tuned classifier with 50 optimization trials
    tuned_clf = TunedTabPFNClassifier(
        n_trials=50,                    # Number of hyperparameter configurations to try
        metric='accuracy',              # Metric to optimize
        categorical_feature_indices=[0, 2],  # Categorical features
        random_state=42                 # For reproducibility
    )

    # Fit will automatically find the best hyperparameters
    tuned_clf.fit(X_train, y_train)

    # Use like any scikit-learn estimator
    y_pred = tuned_clf.predict(X_test)
    ```
"""

from __future__ import annotations

import logging
import warnings
from enum import Enum
from typing import Any, Callable

import numpy as np
import torch
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, check_array
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_X_y

from tabpfn_extensions.hpo.search_space import get_param_grid_hyperopt

# Import TabPFN models from extensions (which handles backend compatibility)
try:
    from tabpfn_extensions.utils import TabPFNClassifier, TabPFNRegressor
except ImportError:
    raise ImportError(
        "TabPFN extensions utils module not found. Please make sure tabpfn_extensions is installed correctly."
    )

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Supported evaluation metrics for TabPFN hyperparameter tuning.

    This enum defines the metrics that can be used to evaluate and select
    the best hyperparameter configuration during optimization.

    Values:
        ACCURACY: Classification accuracy (proportion of correct predictions)
        ROC_AUC: Area under the ROC curve (for binary or multiclass problems)
        F1: F1 score (harmonic mean of precision and recall)
        RMSE: Root mean squared error (regression)
        MSE: Mean squared error (regression)
        MAE: Mean absolute error (regression)
    """

    ACCURACY = "accuracy"
    ROC_AUC = "roc_auc"
    F1 = "f1"
    RMSE = "rmse"
    MSE = "mse"
    MAE = "mae"


class TunedTabPFNBase(BaseEstimator):
    """Base class for tuned TabPFN models with proper categorical handling."""

    def __init__(
        self,
        n_trials: int = 50,
        max_evals: int | None = None,  # Added for backward compatibility
        n_validation_size: float = 0.2,
        metric: MetricType = MetricType.ACCURACY,
        device: str = "auto",
        random_state: int | None = None,
        categorical_feature_indices: list[int] | None = None,
        verbose: bool = True,
        search_space: dict[str, Any] | None = None,
        objective_fn: Callable[[Any, np.ndarray, np.ndarray], float] | None = None,
    ):
        # Handle both n_trials and max_evals parameter names (for backward compatibility)
        self.n_trials = n_trials if max_evals is None else max_evals
        self.max_evals = self.n_trials  # Maintain backward compatibility

        self.n_validation_size = n_validation_size
        self.metric = MetricType(metric)  # Validate metric type
        self.device = device
        self.random_state = random_state
        self.categorical_feature_indices = categorical_feature_indices or []
        self.verbose = verbose
        self.search_space = search_space
        self.objective_fn = objective_fn

    def _setup_data_encoders(
        self,
        X: np.ndarray,
        categorical_feature_indices: list[int] | None = None,
    ):
        """Set up categorical and label encoders."""
        if categorical_feature_indices is not None:
            self.categorical_feature_indices = categorical_feature_indices

        if not self.categorical_feature_indices:
            logger.info(
                "No categorical features specified. Using all features as numeric.",
            )
            self.categorical_feature_indices = []

        # Create categorical feature encoder
        self._cat_encoder = ColumnTransformer(
            transformers=[
                (
                    "categorical",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                    self.categorical_feature_indices,
                ),
            ],
            remainder="passthrough",
            sparse_threshold=0,
        )

    def _optimize(self, X: np.ndarray, y: np.ndarray, task_type: str):
        """Optimize hyperparameters using hyperopt with proper data handling."""
        rng = check_random_state(self.random_state)

        # Set random seeds for reproducibility
        torch.manual_seed(rng.randint(0, 2**31 - 1))
        np.random.seed(rng.randint(0, 2**31 - 1))

        # Fit transformers
        X = self._cat_encoder.fit_transform(X)
        if hasattr(self, "_label_encoder"):
            y = self._label_encoder.transform(y)

        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=self.n_validation_size,
            random_state=rng.randint(0, 2**31 - 1),
            stratify=y if task_type == "multiclass" else None,
        )

        # Use custom search space if provided, otherwise use default
        if hasattr(self, "search_space") and self.search_space is not None:
            # For test with simple search space (just a dict with lists of values)
            custom_space = {}
            for k, v in self.search_space.items():
                if isinstance(v, list):
                    custom_space[k] = hp.choice(k, v)
                elif isinstance(v, (int, float, bool, str)) or v is None:
                    # Handle simple values directly
                    custom_space[k] = v
                else:
                    # Try to handle hyperopt objects directly
                    custom_space[k] = v
            search_space = custom_space
        else:
            search_space = get_param_grid_hyperopt(
                "multiclass" if task_type in ["binary", "multiclass"] else "regression",
            )

        def objective(params):
            inference_config = {
                k.split("/")[-1]: v
                for k, v in params.items()
                if k.startswith("inference_config/") and (_k := k.split("/")[-1])
            }

            # Clean up params
            model_params = {
                k: v for k, v in params.items() if not k.startswith("inference_config/")
            }
            model_params["inference_config"] = inference_config
            # Use device utility for automatic selection
            from tabpfn_extensions.utils import get_device

            model_params["device"] = get_device(self.device)
            model_params["random_state"] = rng.randint(0, 2**31 - 1)

            # Handle special parameters
            n_ensemble_repeats = model_params.pop("n_ensemble_repeats", None)
            if n_ensemble_repeats is not None:
                model_params["n_estimators"] = n_ensemble_repeats

            # Handle model type - skip RF-PFN for now
            model_type = model_params.pop("model_type", "single")
            if model_type == "dt_pfn":
                return {"loss": float("inf"), "status": STATUS_OK}

            try:
                # Create and fit model
                if task_type in ["binary", "multiclass"]:
                    model = TabPFNClassifier(**model_params)
                else:
                    model = TabPFNRegressor(**model_params)

                model.fit(X_train, y_train)

                # Use custom objective function if provided
                if hasattr(self, "objective_fn") and self.objective_fn is not None:
                    # Custom objective should return a negative score (for minimization)
                    score = -self.objective_fn(model, X_val, y_val)
                # Evaluate based on metric
                elif task_type in ["binary", "multiclass"]:
                    if self.metric == MetricType.ACCURACY:
                        score = model.score(X_val, y_val)
                    elif self.metric in [MetricType.ROC_AUC]:
                        from sklearn.metrics import roc_auc_score

                        y_pred = model.predict_proba(X_val)
                        score = roc_auc_score(y_val, y_pred, multi_class="ovr")
                    elif self.metric == MetricType.F1:
                        from sklearn.metrics import f1_score

                        y_pred = model.predict(X_val)
                        score = f1_score(y_val, y_pred, average="weighted")
                else:
                    y_pred = model.predict(X_val)
                    from sklearn.metrics import mean_absolute_error, mean_squared_error

                    if self.metric in [MetricType.RMSE, MetricType.MSE]:
                        score = -mean_squared_error(
                            y_val,
                            y_pred,
                            squared=self.metric == MetricType.MSE,
                        )
                    elif self.metric == MetricType.MAE:
                        score = -mean_absolute_error(y_val, y_pred)

                return {"loss": -score, "status": STATUS_OK, "model": model}

            except (
                ValueError,
                TypeError,
                RuntimeError,
                ImportError,
                torch.cuda.CudaError,
            ) as e:
                if self.verbose:
                    logger.warning(f"Trial failed with error: {e!s}")
                return {"loss": float("inf"), "status": STATUS_OK}

        trials = Trials()
        best = fmin(
            fn=objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=self.n_trials,
            trials=trials,
            verbose=self.verbose,
        )

        # Store results
        self.best_params_ = best
        self.best_score_ = -min(trials.losses())
        self.best_model_ = trials.best_trial["result"].get("model")

        # If all trials failed, create default model
        if self.best_model_ is None:
            warnings.warn(
                "All optimization trials failed. Creating default model.",
                stacklevel=2,
            )
            if task_type in ["binary", "multiclass"]:
                self.best_model_ = TabPFNClassifier(
                    device=self.device,
                    random_state=rng.randint(0, 2**31 - 1),
                )
            else:
                self.best_model_ = TabPFNRegressor(
                    device=self.device,
                    random_state=rng.randint(0, 2**31 - 1),
                )
            self.best_model_.fit(X, y)


class TunedTabPFNClassifier(TunedTabPFNBase, ClassifierMixin):
    """TabPFN Classifier with hyperparameter tuning and proper categorical handling."""

    # TunedTabPFNBase already handles all parameters properly with max_evals for backwards compatibility

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        categorical_feature_indices: list[int] | None = None,
    ) -> TunedTabPFNClassifier:
        # Validate input
        X, y = check_X_y(
            X,
            y,
            force_all_finite="allow-nan",
            dtype=object,
            accept_sparse=False,
        )

        # Store dimensions
        self.n_features_in_ = X.shape[1]

        # Set up encoders
        self._setup_data_encoders(X, categorical_feature_indices)
        self._label_encoder = LabelEncoder()
        y = self._label_encoder.fit_transform(y)

        # Store classes
        self.classes_ = self._label_encoder.classes_

        # Determine task type and optimize
        task_type = "multiclass" if len(self.classes_) > 2 else "binary"
        self._optimize(X, y, task_type)

        # Mark as fitted for sklearn
        self.is_fitted_ = True

        return self

    def __sklearn_is_fitted__(self):
        """Check if the model has been fitted."""
        return (
            hasattr(self, "is_fitted_")
            and self.is_fitted_
            and hasattr(self, "best_model_")
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Simple fit check instead of check_is_fitted to avoid sklearn Tags issue
        if (
            not hasattr(self, "is_fitted_")
            or not self.is_fitted_
            or not hasattr(self, "best_model_")
        ):
            raise ValueError(
                "This TunedTabPFNClassifier instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.",
            )

        X = check_array(X, force_all_finite="allow-nan", dtype=object)
        X = check_array(
            self._cat_encoder.transform(X),
            force_all_finite="allow-nan",
            dtype="numeric",
        )
        return self._label_encoder.inverse_transform(self.best_model_.predict(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Simple fit check instead of check_is_fitted to avoid sklearn Tags issue
        if (
            not hasattr(self, "is_fitted_")
            or not self.is_fitted_
            or not hasattr(self, "best_model_")
        ):
            raise ValueError(
                "This TunedTabPFNClassifier instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.",
            )

        X = check_array(X, force_all_finite="allow-nan", dtype=object)
        X = check_array(
            self._cat_encoder.transform(X),
            force_all_finite="allow-nan",
            dtype="numeric",
        )
        return self.best_model_.predict_proba(X)

    def _more_tags(self):
        return {"allow_nan": True}

    def __sklearn_tags__(self):
        # Create a new dictionary with our tags
        return {"allow_nan": True, "estimator_type": "classifier", "requires_fit": True}


class TunedTabPFNRegressor(TunedTabPFNBase, RegressorMixin):
    """TabPFN Regressor with hyperparameter tuning and proper categorical handling."""

    # TunedTabPFNBase already handles all parameters properly with max_evals for backwards compatibility

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        categorical_feature_indices: list[int] | None = None,
    ) -> TunedTabPFNRegressor:
        # Validate input
        X, y = check_X_y(
            X,
            y,
            force_all_finite="allow-nan",
            dtype=object,
            accept_sparse=False,
        )

        # Store dimensions
        self.n_features_in_ = X.shape[1]

        # Set up encoders
        self._setup_data_encoders(X, categorical_feature_indices)

        # Optimize
        self._optimize(X, y, "regression")

        # Mark as fitted for sklearn
        self.is_fitted_ = True

        return self

    def __sklearn_is_fitted__(self):
        """Check if the model has been fitted."""
        return (
            hasattr(self, "is_fitted_")
            and self.is_fitted_
            and hasattr(self, "best_model_")
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Simple fit check instead of check_is_fitted to avoid sklearn Tags issue
        if (
            not hasattr(self, "is_fitted_")
            or not self.is_fitted_
            or not hasattr(self, "best_model_")
        ):
            raise ValueError(
                "This TunedTabPFNRegressor instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.",
            )

        X = check_array(X, force_all_finite="allow-nan", dtype=object)
        X = check_array(
            self._cat_encoder.transform(X),
            force_all_finite="allow-nan",
            dtype="numeric",
        )
        return self.best_model_.predict(X)

    def _more_tags(self):
        return {"allow_nan": True}

    def __sklearn_tags__(self):
        # Create a new dictionary with our tags
        return {"allow_nan": True, "estimator_type": "regressor", "requires_fit": True}
