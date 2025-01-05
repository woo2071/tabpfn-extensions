import matplotlib.pyplot as plt
import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, load_iris
from tabpfn_extensions import interpretability

from ..examples.interpretability_example import (
    analyze_feature_importance,
    analyze_feature_interactions,
)


@pytest.fixture
def example_data():
    data = load_breast_cancer()
    return data.data[:100], data.target[:100], data.feature_names


class TestFeatureImportance:
    def test_shap_values_shape(self, example_data):
        X, y, feature_names = example_data
        n_samples = 30

        shap_values, _ = analyze_feature_importance(
            X, y, feature_names, n_samples=n_samples, random_state=42
        )

        # Check SHAP values shape
        assert shap_values.shape[0] == n_samples
        assert shap_values.shape[1] == X.shape[1]

    def test_reproducibility(self, example_data):
        X, y, feature_names = example_data
        n_samples = 30

        # Generate two sets of SHAP values with same random state
        shap_values1, _ = analyze_feature_importance(
            X, y, feature_names, n_samples=n_samples, random_state=42
        )
        shap_values2, _ = analyze_feature_importance(
            X, y, feature_names, n_samples=n_samples, random_state=42
        )

        # Check values are identical
        np.testing.assert_array_almost_equal(shap_values1, shap_values2)

    def test_visualization_output(self, example_data):
        X, y, feature_names = example_data

        _, fig = analyze_feature_importance(X, y, feature_names, random_state=42)

        # Check that output is a matplotlib figure
        assert isinstance(fig, plt.Figure)

        # Check that figure contains expected elements
        assert len(fig.axes) > 0  # Should have at least one axis

        # Clean up
        plt.close(fig)


class TestFeatureInteractions:
    def test_invalid_feature_indices(self, example_data):
        X, y, feature_names = example_data

        # Test with invalid feature index
        with pytest.raises(ValueError):
            analyze_feature_interactions(
                X, y, feature_names, feature_idx=(X.shape[1], X.shape[1] + 1)
            )

    def test_visualization_output(self, example_data):
        X, y, feature_names = example_data

        fig = analyze_feature_interactions(
            X, y, feature_names, feature_idx=(0, 1), random_state=42
        )

        # Check that output is a matplotlib figure
        assert isinstance(fig, plt.Figure)

        # Check that figure contains expected elements
        assert len(fig.axes) > 0  # Should have at least one axis

        # Clean up
        plt.close(fig)


def test_end_to_end_workflow(example_data):
    """Test the entire workflow from data input to visualization"""
    X, y, feature_names = example_data

    # Test feature importance analysis
    shap_values, shap_fig = analyze_feature_importance(
        X, y, feature_names, random_state=42
    )
    assert shap_values is not None
    assert isinstance(shap_fig, plt.Figure)
    plt.close(shap_fig)

    # Test feature interaction analysis
    interaction_fig = analyze_feature_interactions(
        X, y, feature_names, feature_idx=(0, 1), random_state=42
    )
    assert isinstance(interaction_fig, plt.Figure)
    plt.close(interaction_fig)


def test_error_handling():
    """Test error handling for invalid inputs"""
    # Create invalid input data
    X = np.random.rand(10, 3)
    y = np.random.randint(0, 2, size=9)  # Mismatched length
    feature_names = ["f1", "f2", "f3"]

    # Test with mismatched X, y dimensions
    with pytest.raises(ValueError):
        analyze_feature_importance(X, y, feature_names)

    # Test with invalid feature names length
    with pytest.raises(ValueError):
        analyze_feature_importance(X, y[:10], feature_names[:-1])
