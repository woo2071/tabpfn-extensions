"""Tests for TabPFN with large datasets functionality.

This test module checks the functionality demonstrated in the large_datasets_example.py
but with very small synthetic datasets to make the tests run quickly.
"""
from __future__ import annotations

import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split

from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor
from tabpfn_extensions.rf_pfn import (
    RandomForestTabPFNClassifier,
    RandomForestTabPFNRegressor,
)


@pytest.fixture
def small_classification_data():
    """Generate a small classification dataset for testing."""
    X, y = make_classification(
        n_samples=30,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        n_classes=2,
        random_state=42,
    )

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42,
    )

    return X_train, X_test, y_train, y_test


@pytest.fixture
def small_regression_data():
    """Generate a small regression dataset for testing."""
    X, y = make_regression(
        n_samples=30,
        n_features=5,
        n_informative=3,
        noise=0.1,
        random_state=42,
    )

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42,
    )

    return X_train, X_test, y_train, y_test


@pytest.mark.requires_tabpfn
def test_tabpfn_subsampled_classifier(small_classification_data):
    """Test TabPFNClassifier with subsampling."""
    X_train, X_test, y_train, y_test = small_classification_data

    # Create model with minimal settings for test
    model = TabPFNClassifier(
        n_estimators=2,  # Minimal ensemble for testing
        ignore_pretraining_limits=True,
        inference_config={
            "SUBSAMPLE_SAMPLES": 10,  # Small number for test speed
        },
    )

    # Fit and predict
    model.fit(X_train, y_train)
    pred_proba = model.predict_proba(X_test)
    preds = model.predict(X_test)

    # Basic checks
    assert pred_proba.shape == (len(X_test), 2)  # Binary classification
    assert len(preds) == len(X_test)
    assert accuracy_score(y_test, preds) > 0.5  # Should be better than random


@pytest.mark.requires_tabpfn
def test_rf_tabpfn_classifier(small_classification_data):
    """Test RandomForestTabPFNClassifier with tree-based approach."""
    X_train, X_test, y_train, y_test = small_classification_data

    # Create base model
    base_model = TabPFNClassifier(
        ignore_pretraining_limits=True,
        inference_config={"SUBSAMPLE_SAMPLES": 10},
    )

    # Create RF model
    model = RandomForestTabPFNClassifier(
        tabpfn=base_model,
        n_estimators=2,  # Minimal for testing
        max_predict_time=5,  # Short timeout
        verbose=0,  # Silent
    )

    # Fit and predict
    model.fit(X_train, y_train)
    pred_proba = model.predict_proba(X_test)
    preds = model.predict(X_test)

    # Basic checks
    assert pred_proba.shape == (len(X_test), 2)  # Binary classification
    assert len(preds) == len(X_test)


@pytest.mark.requires_tabpfn
def test_tabpfn_subsampled_regressor(small_regression_data):
    """Test TabPFNRegressor with subsampling."""
    X_train, X_test, y_train, y_test = small_regression_data

    # Create model with minimal settings for test
    model = TabPFNRegressor(
        n_estimators=2,  # Minimal ensemble for testing
        ignore_pretraining_limits=True,
        inference_config={
            "SUBSAMPLE_SAMPLES": 10,  # Small number for test speed
        },
    )

    # Fit and predict
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # Basic checks
    assert len(preds) == len(X_test)
    assert isinstance(mean_squared_error(y_test, preds), float)


@pytest.mark.requires_tabpfn
def test_rf_tabpfn_regressor(small_regression_data):
    """Test RandomForestTabPFNRegressor with tree-based approach."""
    X_train, X_test, y_train, y_test = small_regression_data

    # Create base model
    base_model = TabPFNRegressor(
        ignore_pretraining_limits=True,
        inference_config={"SUBSAMPLE_SAMPLES": 10},
    )

    # Create RF model
    model = RandomForestTabPFNRegressor(
        tabpfn=base_model,
        n_estimators=2,  # Minimal for testing
        max_predict_time=5,  # Short timeout
        verbose=0,  # Silent
    )

    # Fit and predict
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # Basic checks
    assert len(preds) == len(X_test)
    assert isinstance(mean_squared_error(y_test, preds), float)