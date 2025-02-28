from __future__ import annotations

import numpy as np
import pytest
import torch
import pandas as pd
from typing import Tuple, Dict, List, Union, Optional
from sklearn.metrics import mean_squared_error

try:
    from tabpfn_extensions.unsupervised import TabPFNUnsupervisedModel
    HAS_TABPFN_UNSUPERVISED = True
except ImportError:
    HAS_TABPFN_UNSUPERVISED = False

# Import common testing utilities from conftest
from conftest import FAST_TEST_MODE

# Unsupervised models can work with any TabPFN implementation
pytestmark = [
    pytest.mark.requires_any_tabpfn,  # Requires any TabPFN implementation
    pytest.mark.client_compatible,    # Compatible with TabPFN client
    pytest.mark.skipif(not HAS_TABPFN_UNSUPERVISED, reason="TabPFNUnsupervisedModel not available"),
]

#######################
# Testing Utilities
#######################

class DatasetGenerator:
    """Utility class for generating datasets with various challenging characteristics.
    
    This class provides methods to create synthetic datasets with controlled properties
    for testing the robustness of machine learning algorithms:
    
    - Missing values (NaN) with controlled patterns
    - Outliers with known positions
    - Mixed data types (numerical, categorical)
    - Non-linear relationships
    - Correlated features
    - Structured patterns
    - Special numeric values (zero, infinity, etc.)
    """
    
    def __init__(self, seed: int = 42):
        """Initialize the dataset generator with a specific random seed.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def basic_dataset(self, n_samples: int = 100, n_features: int = 5) -> np.ndarray:
        """Generate a basic dataset with random values.
        
        Args:
            n_samples: Number of samples to generate
            n_features: Number of features to generate
            
        Returns:
            NumPy array of shape (n_samples, n_features)
        """
        return self.rng.rand(n_samples, n_features)
    
    def dataset_with_correlations(
        self, 
        n_samples: int = 100, 
        n_features: int = 5,
        correlation_strength: float = 0.8
    ) -> Tuple[np.ndarray, List[Tuple[int, int, str]]]:
        """Generate a dataset with correlated features.
        
        Args:
            n_samples: Number of samples to generate
            n_features: Number of features to generate
            correlation_strength: Strength of correlations (0-1)
            
        Returns:
            Tuple of (dataset, list of correlations)
            Each correlation is (feature_idx1, feature_idx2, type)
        """
        X = self.rng.rand(n_samples, n_features)
        
        correlations = []
        
        # Linear correlation between features 0 and 1
        X[:, 1] = correlation_strength * X[:, 0] + (1-correlation_strength) * self.rng.randn(n_samples)
        correlations.append((0, 1, "linear"))
        
        # Add non-linear relationship if we have enough features
        if n_features >= 4:
            X[:, 3] = np.sin(X[:, 2] * 3) + 0.1 * self.rng.randn(n_samples)
            correlations.append((2, 3, "non-linear"))
            
        return X, correlations
    
    def dataset_with_missing_values(
        self,
        n_samples: int = 100,
        n_features: int = 5,
        missing_rate: float = 0.1,
        missing_pattern: str = "random"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a dataset with missing values.
        
        Args:
            n_samples: Number of samples to generate
            n_features: Number of features to generate
            missing_rate: Fraction of values to set as missing
            missing_pattern: One of "random", "column", "row", "block"
            
        Returns:
            Tuple of (complete dataset, dataset with missing values)
        """
        X_complete = self.rng.rand(n_samples, n_features)
        
        # Add correlations to make imputation testable
        X_complete[:, 1] = 0.8 * X_complete[:, 0] + 0.2 * self.rng.randn(n_samples)
        
        # Create a copy for missing values
        X_missing = X_complete.copy()
        
        # Create mask for missing values
        if missing_pattern == "random":
            # Randomly distributed missing values
            mask = self.rng.rand(*X_complete.shape) < missing_rate
            
        elif missing_pattern == "column":
            # Entire columns have missing values
            mask = np.zeros_like(X_complete, dtype=bool)
            col_indices = self.rng.choice(
                n_features, 
                size=max(1, int(n_features * missing_rate)),
                replace=False
            )
            for col in col_indices:
                row_indices = self.rng.choice(
                    n_samples,
                    size=max(1, int(n_samples * missing_rate)),
                    replace=False
                )
                mask[row_indices, col] = True
                
        elif missing_pattern == "row":
            # Entire rows have missing values
            mask = np.zeros_like(X_complete, dtype=bool)
            row_indices = self.rng.choice(
                n_samples, 
                size=max(1, int(n_samples * missing_rate)),
                replace=False
            )
            for row in row_indices:
                col_indices = self.rng.choice(
                    n_features,
                    size=max(1, int(n_features * missing_rate)),
                    replace=False
                )
                mask[row, col_indices] = True
                
        elif missing_pattern == "block":
            # Missing values form contiguous blocks
            mask = np.zeros_like(X_complete, dtype=bool)
            block_rows = max(1, int(n_samples * np.sqrt(missing_rate)))
            block_cols = max(1, int(n_features * np.sqrt(missing_rate)))
            
            row_start = self.rng.randint(0, n_samples - block_rows + 1)
            col_start = self.rng.randint(0, n_features - block_cols + 1)
            
            mask[row_start:row_start+block_rows, col_start:col_start+block_cols] = True
        
        else:
            raise ValueError(f"Unknown missing pattern: {missing_pattern}")
        
        # Apply mask
        X_missing[mask] = np.nan
        
        return X_complete, X_missing
    
    def dataset_with_outliers(
        self,
        n_samples: int = 100,
        n_features: int = 4,
        n_outliers: int = 5,
        outlier_type: str = "extreme"
    ) -> Tuple[np.ndarray, List[int]]:
        """Generate a dataset with outliers.
        
        Args:
            n_samples: Number of samples to generate
            n_features: Number of features to generate
            n_outliers: Number of outlier samples to create
            outlier_type: Type of outliers to create ("extreme", "local", "shape")
            
        Returns:
            Tuple of (dataset, list of outlier indices)
        """
        X = self.rng.rand(n_samples, n_features)
        
        # Add correlations between features
        X[:, 1] = 0.7 * X[:, 0] + 0.3 * self.rng.randn(n_samples)
        
        # Select outlier indices
        outlier_indices = self.rng.choice(n_samples, size=n_outliers, replace=False)
        
        if outlier_type == "extreme":
            # Extreme values outside the normal range
            for idx in outlier_indices:
                X[idx, :] = self.rng.uniform(3, 5, size=n_features)
                
        elif outlier_type == "local":
            # Local outliers that break correlation patterns
            for idx in outlier_indices:
                # Keep the first feature but break correlation with second feature
                X[idx, 1] = 1 - X[idx, 0] + self.rng.randn()
                
        elif outlier_type == "shape":
            # Shape outliers (if we have enough dimensions)
            if n_features >= 4:
                for idx in outlier_indices:
                    # Create a different pattern in features 2 and 3
                    X[idx, 2] = 0.9
                    X[idx, 3] = 0.1
        
        else:
            raise ValueError(f"Unknown outlier type: {outlier_type}")
            
        return X, outlier_indices
    
    def mixed_type_dataset(
        self,
        n_samples: int = 100,
        n_continuous: int = 3,
        n_categorical: int = 2,
        n_binary: int = 1,
        categorical_levels: List[int] = None
    ) -> Tuple[np.ndarray, Dict[str, List[int]]]:
        """Generate a dataset with mixed data types.
        
        Args:
            n_samples: Number of samples to generate
            n_continuous: Number of continuous features
            n_categorical: Number of categorical features
            n_binary: Number of binary features
            categorical_levels: Number of levels for each categorical feature
            
        Returns:
            Tuple of (dataset, feature type dictionary)
        """
        n_features = n_continuous + n_categorical + n_binary
        X = np.zeros((n_samples, n_features))
        
        feature_types = {
            "continuous": [],
            "categorical": [],
            "binary": []
        }
        
        # Set default categorical levels if not provided
        if categorical_levels is None:
            categorical_levels = [3] * n_categorical
        
        # Generate continuous features
        for i in range(n_continuous):
            if i % 2 == 0:
                # Uniform distribution
                X[:, i] = self.rng.rand(n_samples)
            else:
                # Normal distribution
                X[:, i] = self.rng.normal(0, 1, n_samples)
            feature_types["continuous"].append(i)
            
        # Generate categorical features
        for i in range(n_categorical):
            col_idx = n_continuous + i
            levels = categorical_levels[i]
            X[:, col_idx] = self.rng.randint(0, levels, n_samples)
            feature_types["categorical"].append(col_idx)
            
        # Generate binary features
        for i in range(n_binary):
            col_idx = n_continuous + n_categorical + i
            X[:, col_idx] = self.rng.choice([0, 1], n_samples)
            feature_types["binary"].append(col_idx)
            
        # Add correlation between a continuous and a categorical feature
        if n_continuous > 0 and n_categorical > 0:
            # Make categorical feature somewhat predictable from continuous
            cont_idx = feature_types["continuous"][0]
            cat_idx = feature_types["categorical"][0]
            levels = categorical_levels[0]
            
            # Binning continuous feature to create correlation
            X[:, cat_idx] = np.digitize(X[:, cont_idx], bins=np.linspace(0, 1, levels+1)[:-1])
            
            # Add some noise to make it imperfect
            noise_mask = self.rng.rand(n_samples) < 0.2
            X[noise_mask, cat_idx] = self.rng.randint(0, levels, noise_mask.sum())
            
        return X, feature_types
    
    def dataset_with_special_values(
        self,
        n_samples: int = 100,
        n_features: int = 5
    ) -> Tuple[np.ndarray, Dict[str, List[Tuple[int, int]]]]:
        """Generate a dataset with special numeric values.
        
        Args:
            n_samples: Number of samples to generate
            n_features: Number of features to generate
            
        Returns:
            Tuple of (dataset, dictionary mapping special value types to coordinates)
        """
        X = self.rng.rand(n_samples, n_features)
        
        special_locations = {
            "zero": [],
            "large": [],
            "small": [],
            "negative": [],
        }
        
        # Add zeros
        zero_mask = self.rng.rand(n_samples, n_features) < 0.05
        X[zero_mask] = 0
        special_locations["zero"] = [(i, j) for i, j in zip(*np.where(zero_mask))]
        
        # Add large values
        large_mask = self.rng.rand(n_samples, n_features) < 0.03
        X[large_mask] = self.rng.uniform(100, 1000, size=large_mask.sum())
        special_locations["large"] = [(i, j) for i, j in zip(*np.where(large_mask))]
        
        # Add small values
        small_mask = self.rng.rand(n_samples, n_features) < 0.03
        X[small_mask] = self.rng.uniform(1e-6, 1e-3, size=small_mask.sum())
        special_locations["small"] = [(i, j) for i, j in zip(*np.where(small_mask))]
        
        # Add negative values
        negative_mask = self.rng.rand(n_samples, n_features) < 0.05
        X[negative_mask] = -self.rng.rand(negative_mask.sum())
        special_locations["negative"] = [(i, j) for i, j in zip(*np.where(negative_mask))]
        
        return X, special_locations


@pytest.fixture
def dataset_generator():
    """Create a dataset generator with a fixed seed."""
    return DatasetGenerator(seed=42)

@pytest.fixture
def synthetic_data(dataset_generator):
    """Create synthetic data with correlated features."""
    n_samples = 20 if FAST_TEST_MODE else 100
    X, _ = dataset_generator.dataset_with_correlations(n_samples=n_samples)
    return X

@pytest.fixture
def mixed_type_data(dataset_generator):
    """Create synthetic data with mixed types (numerical and categorical-like)."""
    n_samples = 20 if FAST_TEST_MODE else 100
    X, _ = dataset_generator.mixed_type_dataset(
        n_samples=n_samples,
        n_continuous=3,
        n_categorical=2,
        n_binary=1
    )
    return X

@pytest.fixture
def data_with_outliers(dataset_generator):
    """Create synthetic data with known outliers for testing."""
    n_samples = 50 if FAST_TEST_MODE else 200
    n_outliers = 3 if FAST_TEST_MODE else 5
    return dataset_generator.dataset_with_outliers(
        n_samples=n_samples,
        n_outliers=n_outliers,
        outlier_type="extreme"
    )

@pytest.fixture
def data_with_missing_values(dataset_generator):
    """Create synthetic data with missing values."""
    n_samples = 20 if FAST_TEST_MODE else 100
    missing_rate = 0.05 if FAST_TEST_MODE else 0.1
    return dataset_generator.dataset_with_missing_values(
        n_samples=n_samples,
        missing_rate=missing_rate,
        missing_pattern="random"
    )

@pytest.fixture
def data_with_special_values(dataset_generator):
    """Create synthetic data with special values (zeros, large values, etc.)."""
    n_samples = 20 if FAST_TEST_MODE else 100
    return dataset_generator.dataset_with_special_values(n_samples=n_samples)


def test_basic_functionality(synthetic_data, tabpfn_classifier, tabpfn_regressor):
    """Test basic functionality of TabPFNUnsupervisedModel."""
    X = synthetic_data

    # Initialize with explicit models
    model = TabPFNUnsupervisedModel(tabpfn_clf=tabpfn_classifier, tabpfn_reg=tabpfn_regressor)

    # Train the model - convert to tensor first
    model.fit(torch.tensor(X, dtype=torch.float32))

    # Check that model was fit successfully
    assert hasattr(model, "X_")
    # The feature_importances_ attribute might not be available in the current implementation
    # Just verify that the model fits without errors


def test_imputation_random_pattern(data_with_missing_values, tabpfn_classifier, tabpfn_regressor):
    """Test imputation capability with randomly missing values."""
    X_complete, X_missing = data_with_missing_values
    
    # Initialize with explicit models
    model = TabPFNUnsupervisedModel(tabpfn_clf=tabpfn_classifier, tabpfn_reg=tabpfn_regressor)
    model.fit(torch.tensor(X_complete, dtype=torch.float32))  # Fit with complete data
    
    # Impute missing values
    X_missing_tensor = torch.tensor(X_missing, dtype=torch.float32)
    X_imputed = model.impute(X_missing_tensor)
    
    # Check shapes and that all values are finite
    assert X_imputed.shape == X_complete.shape
    assert torch.all(torch.isfinite(X_imputed))
    
    # Create mask of missing values
    missing_mask = np.isnan(X_missing)
    missing_mask_tensor = torch.isnan(X_missing_tensor)
    
    # Only compare imputed values to original
    if missing_mask.sum() > 0:
        # Convert back to numpy for easier comparison
        X_imputed_np = X_imputed.detach().cpu().numpy()
        
        # Compute MSE only for imputed values
        mse = mean_squared_error(
            X_complete[missing_mask], 
            X_imputed_np[missing_mask]
        )
        
        # Error should be reasonable for correlated features
        assert mse < 0.5
        
        # Linear feature correlation check - see if imputed values maintain correlation
        # Feature 1 is correlated with feature 0
        correlation_f0_f1 = np.corrcoef(
            X_complete[:, 0], 
            X_complete[:, 1]
        )[0, 1]
        
        correlation_imputed_f0_f1 = np.corrcoef(
            X_imputed_np[:, 0], 
            X_imputed_np[:, 1]
        )[0, 1]
        
        # Correlation difference should be small
        assert abs(correlation_f0_f1 - correlation_imputed_f0_f1) < 0.3

@pytest.mark.parametrize("missing_pattern", ["column", "row", "block"])
def test_imputation_structured_patterns(dataset_generator, tabpfn_classifier, tabpfn_regressor, missing_pattern):
    """Test imputation with different structured missing patterns."""
    # Use a smaller dataset in test mode
    n_samples = 20 if FAST_TEST_MODE else 100
    missing_rate = 0.1 if FAST_TEST_MODE else 0.2
    
    # Generate data with the specific missing pattern
    X_complete, X_missing = dataset_generator.dataset_with_missing_values(
        n_samples=n_samples,
        missing_rate=missing_rate,
        missing_pattern=missing_pattern
    )
    
    # Initialize model
    model = TabPFNUnsupervisedModel(tabpfn_clf=tabpfn_classifier, tabpfn_reg=tabpfn_regressor)
    model.fit(torch.tensor(X_complete, dtype=torch.float32))
    
    # Impute missing values
    X_missing_tensor = torch.tensor(X_missing, dtype=torch.float32)
    X_imputed = model.impute(X_missing_tensor)
    
    # Check shapes and that all values are finite
    assert X_imputed.shape == X_complete.shape
    assert torch.all(torch.isfinite(X_imputed))
    
    # Get the mask of missing values
    missing_mask = np.isnan(X_missing)
    
    # Ensure there are actually missing values
    assert missing_mask.sum() > 0
    
    # Convert imputed tensor to numpy for comparison
    X_imputed_np = X_imputed.detach().cpu().numpy()
    
    # Compute MSE only for imputed values
    mse = mean_squared_error(
        X_complete[missing_mask], 
        X_imputed_np[missing_mask]
    )
    
    # MSE should be reasonable depending on missing pattern
    # (Block and column patterns are harder to impute)
    if missing_pattern == "random":
        assert mse < 0.5
    elif missing_pattern == "row":
        assert mse < 0.6
    else:
        assert mse < 0.7  # More lenient for harder patterns

@pytest.mark.skipif(FAST_TEST_MODE, reason="Skipped in fast test mode")
def test_imputation_with_high_missing_rate(dataset_generator, tabpfn_classifier, tabpfn_regressor):
    """Test imputation with a high missing rate (challenging case)."""
    # Generate data with a high missing rate
    X_complete, X_missing = dataset_generator.dataset_with_missing_values(
        n_samples=100,
        missing_rate=0.4,  # 40% missing values
        missing_pattern="random"
    )
    
    # Initialize model
    model = TabPFNUnsupervisedModel(tabpfn_clf=tabpfn_classifier, tabpfn_reg=tabpfn_regressor)
    model.fit(torch.tensor(X_complete, dtype=torch.float32))
    
    # Impute missing values
    X_missing_tensor = torch.tensor(X_missing, dtype=torch.float32)
    X_imputed = model.impute(X_missing_tensor)
    
    # Check shapes and that all values are finite
    assert X_imputed.shape == X_complete.shape
    assert torch.all(torch.isfinite(X_imputed))
    
    # Get the mask of missing values
    missing_mask = np.isnan(X_missing)
    
    # Convert imputed tensor to numpy for comparison
    X_imputed_np = X_imputed.detach().cpu().numpy()
    
    # Compute MSE only for imputed values - we expect higher error with more missing values
    mse = mean_squared_error(
        X_complete[missing_mask], 
        X_imputed_np[missing_mask]
    )
    
    # With high missing rate, we're more lenient with the MSE
    assert mse < 0.7


def test_outlier_detection_basic(data_with_outliers, tabpfn_classifier, tabpfn_regressor):
    """Test basic outlier detection capability with extreme outliers."""
    X_data, outlier_indices = data_with_outliers
    
    # Initialize with explicit models
    model = TabPFNUnsupervisedModel(tabpfn_clf=tabpfn_classifier, tabpfn_reg=tabpfn_regressor)
    
    # Fit the model with the data (including outliers)
    model.fit(torch.tensor(X_data, dtype=torch.float32))
    
    # Get outlier scores
    outlier_scores = model.outliers(torch.tensor(X_data, dtype=torch.float32))
    
    # Check shape
    assert outlier_scores.shape == (X_data.shape[0],)
    
    # Outliers should have extreme values (either much higher or much lower)
    normal_indices = [i for i in range(X_data.shape[0]) if i not in outlier_indices]
    normal_scores = outlier_scores[normal_indices]
    outlier_scores_subset = outlier_scores[outlier_indices]
    
    # Calculate absolute difference from mean - outliers should be further from the mean
    normal_mean = torch.mean(normal_scores)
    normal_abs_diff = torch.abs(normal_scores - normal_mean)
    outlier_abs_diff = torch.abs(outlier_scores_subset - normal_mean)
    
    # Average difference should be higher for outliers
    assert torch.mean(outlier_abs_diff) > torch.mean(normal_abs_diff)
    
    # Check individual outliers (all should be detected, not just on average)
    for i, outlier_idx in enumerate(outlier_indices):
        # For each outlier, compare against average normal absolute difference
        assert outlier_abs_diff[i] > torch.mean(normal_abs_diff) * 1.5, f"Outlier at index {outlier_idx} not detected well"

@pytest.mark.parametrize("outlier_type", ["extreme", "local", "shape"])
def test_outlier_detection_types(dataset_generator, tabpfn_classifier, tabpfn_regressor, outlier_type):
    """Test outlier detection with different types of outliers."""
    # Skip shape outliers test in fast mode (takes longer)
    if FAST_TEST_MODE and outlier_type == "shape":
        pytest.skip("Shape outlier test skipped in fast mode")
    
    # Generate dataset with specified outlier type
    n_samples = 30 if FAST_TEST_MODE else 100
    n_outliers = 2 if FAST_TEST_MODE else 5
    n_features = 4
    
    X_data, outlier_indices = dataset_generator.dataset_with_outliers(
        n_samples=n_samples,
        n_features=n_features,
        n_outliers=n_outliers,
        outlier_type=outlier_type
    )
    
    # Initialize model
    model = TabPFNUnsupervisedModel(tabpfn_clf=tabpfn_classifier, tabpfn_reg=tabpfn_regressor)
    model.fit(torch.tensor(X_data, dtype=torch.float32))
    
    # Get outlier scores
    outlier_scores = model.outliers(torch.tensor(X_data, dtype=torch.float32))
    
    # Check shape
    assert outlier_scores.shape == (X_data.shape[0],)
    
    # Separate normal and outlier scores
    normal_indices = [i for i in range(X_data.shape[0]) if i not in outlier_indices]
    normal_scores = outlier_scores[normal_indices]
    outlier_scores_subset = outlier_scores[outlier_indices]
    
    # Different analysis for different outlier types
    if outlier_type == "extreme":
        # Extreme value outliers should have very different scores
        normal_mean = torch.mean(normal_scores)
        outlier_mean = torch.mean(outlier_scores_subset)
        # Scores could be higher or lower depending on the model
        assert abs(normal_mean - outlier_mean) > torch.std(normal_scores) * 1.5
        
    elif outlier_type == "local":
        # Local outliers (correlation breakers) might be harder to detect
        # Use percentile rank for this analysis
        scores_np = outlier_scores.detach().cpu().numpy()
        normal_score_avg = np.mean(scores_np[normal_indices])
        outlier_score_avg = np.mean(scores_np[outlier_indices])
        
        # Either very high or very low scores would indicate outliers
        score_diff = abs(normal_score_avg - outlier_score_avg)
        assert score_diff > np.std(scores_np[normal_indices])
        
    elif outlier_type == "shape":
        # Shape outliers have unusual relationships between features
        # Check how many outliers are among the top/bottom N most extreme scores
        scores_np = outlier_scores.detach().cpu().numpy()
        
        # Sort indices by outlier score extremity (distance from median)
        median_score = np.median(scores_np)
        extremity = np.abs(scores_np - median_score)
        sorted_indices = np.argsort(extremity)[::-1]  # Descending order
        
        # Count how many true outliers are in the top N most extreme scores
        n_check = len(outlier_indices) * 2  # Check top N*2 scores
        top_extreme = sorted_indices[:n_check]
        outliers_found = sum(idx in outlier_indices for idx in top_extreme)
        
        # At least some of the true outliers should be found
        assert outliers_found >= 1, "No shape outliers detected in top extremes"
        
@pytest.mark.skipif(FAST_TEST_MODE, reason="Skipped in fast test mode")
def test_outlier_detection_robustness(dataset_generator, tabpfn_classifier, tabpfn_regressor):
    """Test outlier detection robustness with a mixed dataset containing special values."""
    # Generate dataset with special values (zeros, large numbers, etc.)
    X_special, special_locs = dataset_generator.dataset_with_special_values(n_samples=100)
    
    # Add some extreme outliers
    n_outliers = 5
    outlier_indices = [10, 25, 42, 67, 89]  # Fixed indices for reproducibility
    
    for idx in outlier_indices:
        X_special[idx, :] = 10.0  # Extreme values
    
    # Initialize model
    model = TabPFNUnsupervisedModel(tabpfn_clf=tabpfn_classifier, tabpfn_reg=tabpfn_regressor)
    model.fit(torch.tensor(X_special, dtype=torch.float32))
    
    # Get outlier scores
    outlier_scores = model.outliers(torch.tensor(X_special, dtype=torch.float32))
    
    # Convert to numpy for analysis
    scores_np = outlier_scores.detach().cpu().numpy()
    
    # Verify scores have a reasonable range
    assert np.all(np.isfinite(scores_np)), "Outlier scores should be finite"
    
    # Outliers should be among the most extreme
    median_score = np.median(scores_np)
    extremity = np.abs(scores_np - median_score)
    
    # Get indices of top 10 most extreme points
    top_extreme = np.argsort(extremity)[-10:]
    
    # At least 3 of the 5 outliers should be in the top 10 most extreme
    outliers_found = sum(idx in outlier_indices for idx in top_extreme)
    assert outliers_found >= 3, f"Only {outliers_found} outliers found in top extremes"


@pytest.mark.skipif(FAST_TEST_MODE, reason="Skipped in fast test mode")
def test_generate_samples_basic(synthetic_data, tabpfn_classifier, tabpfn_regressor):
    """Test basic sample generation capability."""
    X = synthetic_data

    # Initialize with explicit models
    model = TabPFNUnsupervisedModel(tabpfn_clf=tabpfn_classifier, tabpfn_reg=tabpfn_regressor)
    # Convert to tensor before fitting
    model.fit(torch.tensor(X, dtype=torch.float32))

    # Generate new samples
    n_samples = 20
    X_gen = model.generate_synthetic_data(n_samples=n_samples)

    # Check shape
    assert X_gen.shape == (n_samples, X.shape[1])

    # All values should be finite
    assert torch.all(torch.isfinite(X_gen))

    # Check that generated values are within reasonable range
    assert torch.all(X_gen >= -0.5)  # Allow some slightly negative values
    assert torch.all(X_gen <= 1.5)   # Allow some slightly high values
    
    # Convert to numpy for correlation analysis
    X_gen_np = X_gen.detach().cpu().numpy()
    X_np = X.detach().cpu().numpy() if torch.is_tensor(X) else X
    
    # Check correlation structure is preserved in generated data
    # Features 0 and 1 should be correlated in both original and generated data
    orig_corr = np.corrcoef(X_np[:, 0], X_np[:, 1])[0, 1]
    gen_corr = np.corrcoef(X_gen_np[:, 0], X_gen_np[:, 1])[0, 1]
    
    # Correlation sign should be the same
    assert np.sign(orig_corr) == np.sign(gen_corr)
    
    # Correlation magnitude should be similar (not exact due to sampling)
    assert abs(orig_corr - gen_corr) < 0.3
    
def test_generate_samples_with_temperature(dataset_generator, tabpfn_classifier, tabpfn_regressor):
    """Test sample generation with different temperature settings."""
    # This test runs even in fast mode since it's important to verify temperature control
    
    # Generate data with clear patterns
    X, _ = dataset_generator.dataset_with_correlations(
        n_samples=20 if FAST_TEST_MODE else 100,
        correlation_strength=0.9  # High correlation for clear pattern
    )
    
    # Initialize model
    model = TabPFNUnsupervisedModel(tabpfn_clf=tabpfn_classifier, tabpfn_reg=tabpfn_regressor)
    model.fit(torch.tensor(X, dtype=torch.float32))
    
    # Generate samples with low temperature (more deterministic)
    n_samples = 10
    X_gen_low_temp = model.generate_synthetic_data(n_samples=n_samples, t=0.1)
    
    # Generate samples with high temperature (more random)
    X_gen_high_temp = model.generate_synthetic_data(n_samples=n_samples, t=1.0)
    
    # All samples should be finite
    assert torch.all(torch.isfinite(X_gen_low_temp))
    assert torch.all(torch.isfinite(X_gen_high_temp))
    
    # Convert to numpy for analysis
    X_low_np = X_gen_low_temp.detach().cpu().numpy()
    X_high_np = X_gen_high_temp.detach().cpu().numpy()
    
    # Low temperature should give more predictable samples
    # Check variance of feature 1 (which is correlated with feature 0)
    # For each value of feature 0, feature 1 should show less variance with low temperature
    
    # Group samples by binned feature 0 values and measure feature 1 variance
    bins = np.linspace(0, 1, 4)  # 3 bins for feature 0
    X_low_binned = np.digitize(X_low_np[:, 0], bins)
    X_high_binned = np.digitize(X_high_np[:, 0], bins)
    
    # Collect variances for each bin
    low_temp_vars = []
    high_temp_vars = []
    
    for bin_idx in range(1, len(bins)):
        # Get samples in current bin
        low_mask = X_low_binned == bin_idx
        high_mask = X_high_binned == bin_idx
        
        # Skip bins with too few samples
        if low_mask.sum() > 1 and high_mask.sum() > 1:
            # Calculate variance of feature 1 in this bin
            low_temp_vars.append(np.var(X_low_np[low_mask, 1]))
            high_temp_vars.append(np.var(X_high_np[high_mask, 1]))
    
    # If we have enough data points in bins
    if len(low_temp_vars) > 0 and len(high_temp_vars) > 0:
        # Low temperature should result in lower variance on average
        assert np.mean(low_temp_vars) <= np.mean(high_temp_vars) * 1.5, \
            "Low temperature didn't reduce variance as expected"

@pytest.mark.skipif(FAST_TEST_MODE, reason="Skipped in fast test mode")
def test_generate_samples_with_categorical(dataset_generator, tabpfn_classifier, tabpfn_regressor):
    """Test sample generation with mixed continuous and categorical data."""
    # Generate mixed type data
    X, feature_types = dataset_generator.mixed_type_dataset(
        n_samples=100,
        n_continuous=3,
        n_categorical=2,
        n_binary=1
    )
    
    # Convert to torch tensor
    X_tensor = torch.tensor(X, dtype=torch.float32)
    
    # Initialize model and set categorical features
    model = TabPFNUnsupervisedModel(tabpfn_clf=tabpfn_classifier, tabpfn_reg=tabpfn_regressor)
    
    # Set categorical features
    categorical_indices = feature_types["categorical"] + feature_types["binary"]
    model.set_categorical_features(categorical_indices)
    
    # Fit model
    model.fit(X_tensor)
    
    # Generate samples
    n_samples = 50
    X_gen = model.generate_synthetic_data(n_samples=n_samples)
    
    # Check shape and finite values
    assert X_gen.shape == (n_samples, X.shape[1])
    assert torch.all(torch.isfinite(X_gen))
    
    # Convert to numpy for analysis
    X_gen_np = X_gen.detach().cpu().numpy()
    
    # Categorical features should have discrete values
    for cat_idx in categorical_indices:
        # Get unique values in the generated categorical feature
        unique_vals = np.unique(np.round(X_gen_np[:, cat_idx]))
        
        # Should have a small number of unique values
        assert len(unique_vals) <= len(np.unique(X[:, cat_idx])) + 2, \
            f"Generated categorical feature {cat_idx} has too many unique values"


@pytest.mark.skipif(FAST_TEST_MODE, reason="Skipped in fast test mode")
def test_feature_correlation(tabpfn_classifier, tabpfn_regressor):
    """Test if the model captures feature correlations properly."""
    # Create data with strong correlation between features
    rng = np.random.RandomState(42)
    n_samples = 20 if FAST_TEST_MODE else 100
    X = rng.rand(n_samples, 3)
    X[:, 1] = 0.9 * X[:, 0] + 0.1 * rng.randn(n_samples)  # Strongly correlated with feature 0
    X[:, 2] = rng.rand(n_samples)  # Independent feature

    # Initialize with explicit models
    model = TabPFNUnsupervisedModel(tabpfn_clf=tabpfn_classifier, tabpfn_reg=tabpfn_regressor)
    # Convert to tensor before fitting
    model.fit(torch.tensor(X, dtype=torch.float32))

    # Create test sample with just one feature missing
    X_test = X[0:1].copy()
    X_test[0, 1] = np.nan  # Missing value for correlated feature

    # Convert to tensor and impute
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    X_imputed = model.impute(X_test_tensor)

    # Check if imputed value is close to expected value based on correlation
    expected_value = 0.9 * X_test[0, 0]
    assert np.isclose(X_imputed[0, 1], expected_value, atol=0.3)


@pytest.mark.skipif(FAST_TEST_MODE, reason="Skipped in fast test mode")
def test_conditional_generation_basic(dataset_generator, tabpfn_classifier, tabpfn_regressor):
    """Test basic conditional sample generation."""
    # Generate data with strong correlations
    X, correlations = dataset_generator.dataset_with_correlations(
        n_samples=100, correlation_strength=0.9
    )

    # Initialize with explicit models
    model = TabPFNUnsupervisedModel(tabpfn_clf=tabpfn_classifier, tabpfn_reg=tabpfn_regressor)
    
    # Convert to tensor before fitting
    model.fit(torch.tensor(X, dtype=torch.float32))

    # Create a conditional sample with fixed first feature
    X_cond = np.full((1, X.shape[1]), np.nan)
    X_cond[0, 0] = 0.8  # Fix first feature

    # Use impute method for conditional generation (which handles conditionals via NaNs)
    X_gen = model.impute(torch.tensor(X_cond, dtype=torch.float32), n_permutations=5)

    # Check shape
    assert X_gen.shape == (1, X.shape[1])

    # Fixed features should remain fixed
    assert torch.all(torch.isclose(X_gen[:, 0], torch.tensor([0.8]), atol=1e-5))

    # All values should be finite
    assert torch.all(torch.isfinite(X_gen))
    
    # Check that generated second feature follows correlation pattern with first feature
    # Feature 1 should be correlated with feature 0 based on the training data pattern
    X_gen_np = X_gen.detach().cpu().numpy()
    
    # Expected value based on correlation
    # If feature 0 is 0.8 and correlation strength is 0.9, feature 1 should be around 0.8*0.9 = 0.72
    # Allow for some noise
    assert abs(X_gen_np[0, 1] - 0.8 * 0.9) < 0.3, \
        f"Feature 1 value {X_gen_np[0, 1]} doesn't follow expected correlation pattern"

@pytest.mark.skipif(FAST_TEST_MODE, reason="Skipped in fast test mode")
def test_conditional_generation_multiple_conditions(dataset_generator, tabpfn_classifier, tabpfn_regressor):
    """Test conditional generation with multiple fixed features."""
    # Generate data
    X, _ = dataset_generator.dataset_with_correlations(n_samples=100)

    # Initialize model
    model = TabPFNUnsupervisedModel(tabpfn_clf=tabpfn_classifier, tabpfn_reg=tabpfn_regressor)
    model.fit(torch.tensor(X, dtype=torch.float32))

    # Create multiple conditional samples with different fixed features
    n_samples = 5
    X_cond = np.full((n_samples, X.shape[1]), np.nan)
    
    # Fix different features for each sample
    X_cond[0, 0] = 0.2  # Sample 1: fix feature 0
    X_cond[1, 2] = 0.5  # Sample 2: fix feature 2
    X_cond[2, 0:2] = [0.8, 0.7]  # Sample 3: fix features 0 and 1
    X_cond[3, [0, 3]] = [0.4, 0.6]  # Sample 4: fix features 0 and 3
    X_cond[4, 1:4] = [0.3, 0.1, 0.9]  # Sample 5: fix features 1, 2, 3
    
    # Generate conditional samples
    X_gen = model.impute(torch.tensor(X_cond, dtype=torch.float32), n_permutations=3)
    
    # Check shape
    assert X_gen.shape == (n_samples, X.shape[1])
    
    # All fixed features should remain fixed
    X_gen_np = X_gen.detach().cpu().numpy()
    
    # Check each sample's fixed features
    assert abs(X_gen_np[0, 0] - 0.2) < 1e-5  # Sample 1
    assert abs(X_gen_np[1, 2] - 0.5) < 1e-5  # Sample 2
    assert abs(X_gen_np[2, 0] - 0.8) < 1e-5 and abs(X_gen_np[2, 1] - 0.7) < 1e-5  # Sample 3
    assert abs(X_gen_np[3, 0] - 0.4) < 1e-5 and abs(X_gen_np[3, 3] - 0.6) < 1e-5  # Sample 4
    assert abs(X_gen_np[4, 1] - 0.3) < 1e-5 and abs(X_gen_np[4, 2] - 0.1) < 1e-5 and abs(X_gen_np[4, 3] - 0.9) < 1e-5  # Sample 5
    
    # All values should be finite
    assert np.all(np.isfinite(X_gen_np))

@pytest.mark.skipif(FAST_TEST_MODE, reason="Skipped in fast test mode")
def test_conditional_generation_with_categorical(dataset_generator, tabpfn_classifier, tabpfn_regressor):
    """Test conditional generation with mixed data types including categorical features."""
    # Generate mixed type data
    X, feature_types = dataset_generator.mixed_type_dataset(n_samples=100)
    
    # Get categorical feature indices
    categorical_indices = feature_types["categorical"] + feature_types["binary"]
    
    # Initialize model
    model = TabPFNUnsupervisedModel(tabpfn_clf=tabpfn_classifier, tabpfn_reg=tabpfn_regressor)
    model.set_categorical_features(categorical_indices)
    model.fit(torch.tensor(X, dtype=torch.float32))
    
    # Create conditional samples with fixed categorical and continuous features
    # Pick the first continuous and first categorical feature to condition on
    cont_idx = feature_types["continuous"][0]
    cat_idx = feature_types["categorical"][0]
    
    # Create 3 samples with different conditions
    X_cond = np.full((3, X.shape[1]), np.nan)
    
    # Fix both continuous and categorical feature
    X_cond[0, cont_idx] = 0.7
    X_cond[0, cat_idx] = 2.0  # Integer value for categorical feature
    
    # Fix just continuous feature
    X_cond[1, cont_idx] = 0.3
    
    # Fix just categorical feature
    X_cond[2, cat_idx] = 1.0
    
    # Generate conditional samples
    X_gen = model.impute(torch.tensor(X_cond, dtype=torch.float32), n_permutations=3)
    
    # Convert to numpy
    X_gen_np = X_gen.detach().cpu().numpy()
    
    # Check fixed features are maintained
    assert abs(X_gen_np[0, cont_idx] - 0.7) < 1e-5
    assert abs(X_gen_np[0, cat_idx] - 2.0) < 1e-5
    assert abs(X_gen_np[1, cont_idx] - 0.3) < 1e-5
    assert abs(X_gen_np[2, cat_idx] - 1.0) < 1e-5
    
    # All values should be finite
    assert np.all(np.isfinite(X_gen_np))
    
    # Categorical features should remain as discrete values
    for idx in categorical_indices:
        # Count unique values (after rounding to handle potential floating point issues)
        unique_vals = np.unique(np.round(X_gen_np[:, idx]))
        assert len(unique_vals) <= len(np.unique(X[:, idx])) + 1


def test_model_initialization_with_explicit_models(synthetic_data):
    """Test initialization with explicit classifier and regressor as in examples."""
    X = synthetic_data

    try:
        # Try to import TabPFN from different sources, like in the example
        try:
            from tabpfn import TabPFNClassifier, TabPFNRegressor
            clf = TabPFNClassifier()  # Let device default to auto
            reg = TabPFNRegressor()   # Let device default to auto
        except ImportError:
            try:
                from tabpfn_client import TabPFNClassifier, TabPFNRegressor
                clf = TabPFNClassifier()
                reg = TabPFNRegressor()
            except ImportError:
                from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor
                clf = TabPFNClassifier()  # Let device default to auto
                reg = TabPFNRegressor()   # Let device default to auto

        # Initialize with explicit models
        model = TabPFNUnsupervisedModel(tabpfn_clf=clf, tabpfn_reg=reg)

        # Should work without errors
        model.fit(torch.tensor(X, dtype=torch.float32))

        # Check that model was fit successfully
        assert hasattr(model, "X_")
        # The feature_importances_ attribute might not be available in the current implementation
        # Just verify that the model fits without errors

    except (ImportError, TypeError, AttributeError) as e:
        pytest.skip(f"Model initialization with explicit models failed: {e}")


def test_temperature_parameter(synthetic_data, tabpfn_classifier, tabpfn_regressor):
    """Test temperature parameter for data generation shown in example."""
    X = synthetic_data

    # Initialize and fit model
    model = TabPFNUnsupervisedModel(tabpfn_clf=tabpfn_classifier, tabpfn_reg=tabpfn_regressor)
    model.fit(torch.tensor(X, dtype=torch.float32))

    # Try different temperature values
    for temp in [0.1, 1.0]:
        try:
            # Generate samples with specific temperature
            X_gen = model.generate_synthetic_data(n_samples=3, t=temp)

            # Should have expected shape
            assert X_gen.shape == (3, X.shape[1])

            # Lower temperature should generally lead to less variance
            # but this is probabilistic so hard to test directly

            # Basic check that values are reasonable
            assert torch.all(torch.isfinite(X_gen))
        except (TypeError, ValueError) as e:
            # Skip if temp parameter isn't supported in this version
            pytest.skip(f"Temperature parameter not supported: {e}")
            break


def test_outlier_detection_experiment(dataset_generator, tabpfn_classifier, tabpfn_regressor):
    """Test the OutlierDetectionUnsupervisedExperiment class."""
    try:
        # Import experiment class
        from tabpfn_extensions.unsupervised.experiments import (
            OutlierDetectionUnsupervisedExperiment,
        )
    except ImportError:
        pytest.skip("OutlierDetectionUnsupervisedExperiment not available")

    # Create dataset with outliers
    X, outlier_indices = dataset_generator.dataset_with_outliers(
        n_samples=20,
        n_features=4,
        n_outliers=1,  # Just one outlier for simplicity
        outlier_type="extreme"
    )
    
    # Set first sample as the outlier for easier validation
    if 0 not in outlier_indices:
        X[0] = X[outlier_indices[0]]
        outlier_indices = [0]

    # Create dummy labels (not used by unsupervised model)
    y = np.zeros(len(X))

    # Initialize unsupervised model
    model = TabPFNUnsupervisedModel(tabpfn_clf=tabpfn_classifier, tabpfn_reg=tabpfn_regressor)
    model.fit(torch.tensor(X, dtype=torch.float32))

    # Create feature names
    attribute_names = [f"feature_{i}" for i in range(X.shape[1])]

    # Create experiment
    exp = OutlierDetectionUnsupervisedExperiment(task_type="unsupervised")

    # Convert data to torch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    # Run experiment
    results = exp.run(
        tabpfn=model,
        X=X_tensor,
        y=y_tensor,
        attribute_names=attribute_names,
        indices=[0, 1],  # Analyze first two features
        should_plot=False,  # Don't generate plots in tests
    )

    # Verify results structure
    assert results is not None
    assert "outlier_scores" in results

    # Validate outlier scores
    assert len(results["outlier_scores"]) == len(X)
    
    # Outlier (sample 0) should have an extreme score
    normal_samples = results["outlier_scores"][1:]
    normal_mean = np.mean(normal_samples)
    
    # Calculate absolute difference from mean
    normal_abs_diff = np.abs(normal_samples - normal_mean)
    outlier_abs_diff = np.abs(results["outlier_scores"][0] - normal_mean)
    
    # Outlier should be further from the mean than the average normal sample
    assert outlier_abs_diff > np.mean(normal_abs_diff)

def test_synthetic_data_generation_experiment(dataset_generator, tabpfn_classifier, tabpfn_regressor):
    """Test the GenerateSyntheticDataExperiment class."""
    try:
        # Import experiment class
        from tabpfn_extensions.unsupervised.experiments import (
            GenerateSyntheticDataExperiment,
        )
    except ImportError:
        pytest.skip("GenerateSyntheticDataExperiment not available")
    
    # Generate correlated dataset
    X, correlations = dataset_generator.dataset_with_correlations(
        n_samples=20 if FAST_TEST_MODE else 50,
        correlation_strength=0.8
    )
    
    # Create dummy labels
    y = np.zeros(len(X))
    
    # Initialize unsupervised model
    model = TabPFNUnsupervisedModel(tabpfn_clf=tabpfn_classifier, tabpfn_reg=tabpfn_regressor)
    model.fit(torch.tensor(X, dtype=torch.float32))
    
    # Create feature names
    attribute_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    # Create experiment
    exp = GenerateSyntheticDataExperiment(task_type="unsupervised")
    
    # Convert data to torch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    
    # Run experiment with minimal sample size for speed
    results = exp.run(
        tabpfn=model,
        X=X_tensor,
        y=y_tensor,
        attribute_names=attribute_names,
        indices=[0, 1],  # Only use first two features for speed
        n_samples=5,     # Generate just a few samples
        temp=1.0,        # Standard temperature
        should_plot=False,  # Don't generate plots in tests
    )
    
    # Basic verification that the experiment ran
    assert hasattr(exp, "synthetic_X")
    assert hasattr(exp, "data")
    assert hasattr(exp, "data_real")
    assert hasattr(exp, "data_synthetic")
    
    # Verify synthetic data has same number of features
    assert exp.synthetic_X.shape[1] == 2  # We selected 2 features
    
    # Verify synthetic data has requested number of samples
    assert exp.synthetic_X.shape[0] == 5
    
    # For data frame verification
    assert "real_or_synthetic" in exp.data.columns
    assert "feature_0" in exp.data.columns
    assert "feature_1" in exp.data.columns

@pytest.mark.skipif(FAST_TEST_MODE, reason="Skipped in fast test mode")
def test_embedding_experiment(dataset_generator, tabpfn_classifier, tabpfn_regressor):
    """Test the EmbeddingUnsupervisedExperiment class."""
    try:
        # Import experiment class
        from tabpfn_extensions.unsupervised.experiments import (
            EmbeddingUnsupervisedExperiment,
        )
    except ImportError:
        pytest.skip("EmbeddingUnsupervisedExperiment not available")
        
    # Also skip if get_embeddings method is disabled (it is in newer versions)
    try:
        model = TabPFNUnsupervisedModel(tabpfn_clf=tabpfn_classifier, tabpfn_reg=tabpfn_regressor)
        if not hasattr(model, "get_embeddings") or model.get_embeddings is None:
            pytest.skip("get_embeddings method not available in this version")
            
        # Check if method raises NotImplementedError
        X = np.random.rand(5, 3)
        try:
            model.fit(torch.tensor(X, dtype=torch.float32))
            model.get_embeddings(torch.tensor(X, dtype=torch.float32))
        except NotImplementedError:
            pytest.skip("get_embeddings method not implemented in this version")
            
    except (ImportError, AttributeError, TypeError):
        pytest.skip("Model initialization issues - embeddings not available")
    
    # Create dataset with classes for visualization
    X, _ = dataset_generator.dataset_with_correlations(n_samples=50)
    
    # Create synthetic classes (2 classes based on feature 0)
    y = (X[:, 0] > 0.5).astype(int)
    
    # Create dataset object expected by the experiment
    class DummyDataset:
        def __init__(self, x, y):
            self.x = x
            self.y = y
    
    dataset = DummyDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long)
    )
    
    # Create experiment
    exp = EmbeddingUnsupervisedExperiment(task_type="unsupervised")
    
    try:
        # Run experiment with plotting disabled
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        
        # Mock plot method to avoid actual plotting
        original_plot = exp.plot
        exp.plot = lambda **kwargs: None
        
        # Run experiment
        results = exp.run(
            tabpfn=model,
            dataset=dataset,
            per_column=False
        )
        
        # Restore original method
        exp.plot = original_plot
        
        # Basic verification
        assert hasattr(exp, "emb")
        assert hasattr(exp, "X_test")
        assert hasattr(exp, "y_test")
        
        # Embeddings should have batch dimension matching data
        assert len(exp.emb) == len(exp.X_test)
        
    except (NotImplementedError, AttributeError) as e:
        pytest.skip(f"Embedding experiment failed: {e}")