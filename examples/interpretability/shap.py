import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from tabpfn_extensions import interpretability
from tabpfn_extensions import TabPFNClassifier
from typing import List, Optional, Tuple


def analyze_feature_importance(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_samples: int = 100,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, plt.Figure]:
    """
    Analyze feature importance using SHAP values and create visualization.

    Args:
        X: Input features array
        y: Target labels array
        feature_names: List of feature names
        n_samples: Number of samples to use for analysis
        random_state: Random state for reproducibility

    Returns:
        Tuple containing SHAP values array and matplotlib figure
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state
    )

    # Initialize and train model
    clf = TabPFNClassifier()
    clf.fit(X_train, y_train)

    # Calculate SHAP values
    shap_values = interpretability.shap.get_shap_values(
        model=clf,
        X=X_test[:n_samples],
        attribute_names=feature_names,
        algorithm="permutation",
    )

    # Create visualization
    fig = interpretability.shap.plot_shap(shap_values)

    return shap_values, fig


# Load example dataset
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names

# Analyze feature importance
shap_values, shap_fig = analyze_feature_importance(X, y, feature_names, random_state=42)
shap_fig.savefig("feature_importance.png")
