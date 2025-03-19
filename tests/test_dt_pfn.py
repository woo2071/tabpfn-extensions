"""Tests for the TabPFN-based Decision Tree implementation.

This file tests the Decision Tree TabPFN implementations in tabpfn_extensions.rf_pfn.
"""

from __future__ import annotations

import pytest

from tabpfn_extensions.rf_pfn.sklearn_based_decision_tree_tabpfn import (
    DecisionTreeTabPFNClassifier,
    DecisionTreeTabPFNRegressor,
)
from test_base_tabpfn import BaseClassifierTests, BaseRegressorTests


class TestDecisionTreeClassifier(BaseClassifierTests):
    """Test DecisionTreeTabPFNClassifier using the BaseClassifierTests framework."""

    @pytest.fixture
    def estimator(self, tabpfn_classifier):
        """Provide a TabPFN-based DecisionTreeClassifier as the estimator."""
        return DecisionTreeTabPFNClassifier(
            tabpfn=tabpfn_classifier,
            max_depth=3,  # Shallow tree for speed
            random_state=42,
        )

    # Skip pandas and text feature tests as they're not fully supported by DecisionTreeTabPFN
    @pytest.mark.skip(
        reason="DecisionTreeTabPFN doesn't fully support pandas DataFrames",
    )
    def test_with_pandas(self, estimator, pandas_classification_data):
        pass

    @pytest.mark.skip(reason="DecisionTreeTabPFN doesn't fully support text features")
    def test_with_text_features(self, estimator, dataset_generator):
        pass


class TestAdaptiveDecisionTreeClassifier(TestDecisionTreeClassifier):
    @pytest.fixture
    def estimator(self, tabpfn_classifier):
        """Provide a TabPFN-based DecisionTreeClassifier as the estimator."""
        return DecisionTreeTabPFNClassifier(
            tabpfn=tabpfn_classifier,
            max_depth=3,  # Shallow tree for speed
            random_state=42,
        )


class TestDecisionTreeRegressor(BaseRegressorTests):
    """Test DecisionTreeTabPFNRegressor using the BaseRegressorTests framework."""

    @pytest.fixture
    def estimator(self, tabpfn_regressor):
        """Provide a TabPFN-based DecisionTreeRegressor as the estimator."""
        return DecisionTreeTabPFNRegressor(
            tabpfn=tabpfn_regressor,
            max_depth=3,  # Shallow tree for speed
            random_state=42,
        )

    # Skip pandas and text feature tests as they're not fully supported by DecisionTreeTabPFN
    @pytest.mark.skip(
        reason="DecisionTreeTabPFN doesn't fully support pandas DataFrames",
    )
    def test_with_pandas(self, estimator, pandas_regression_data):
        pass

    @pytest.mark.skip(reason="DecisionTreeTabPFN doesn't fully support text features")
    def test_with_text_features(self, estimator, dataset_generator):
        pass


class TestAdaptiveDecisionTreeRegressor(TestDecisionTreeRegressor):
    @pytest.fixture
    def estimator(self, tabpfn_regressor):
        """Provide a TabPFN-based DecisionTreeRegressor as the estimator."""
        return DecisionTreeTabPFNRegressor(
            tabpfn=tabpfn_regressor,
            max_depth=3,  # Shallow tree for speed
            random_state=42,
            adaptive_tree=True,
        )
