import numpy as np
import pandas as pd
import pytest
from conftest import FAST_TEST_MODE

#######################
# Testing Utilities
#######################

# TODO: Also test text data input

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

    def basic_dataset(
        self, 
        n_samples: int = 100, 
        n_features: int = 5, 
        as_pandas: bool = False,
        column_names: list[str] = None,
        return_feature_names: bool = False
    ) -> np.ndarray:
        """Generate a basic dataset with random values.

        Args:
            n_samples: Number of samples to generate
            n_features: Number of features to generate
            as_pandas: If True, return a pandas DataFrame instead of numpy array
            column_names: Custom column names for pandas DataFrame output
            return_feature_names: If True, return feature names as the third element

        Returns:
            NumPy array of shape (n_samples, n_features) or pandas DataFrame
            If return_feature_names is True, also returns a list of feature names
        """
        X = self.rng.rand(n_samples, n_features)
        
        # Generate feature names if not provided
        if column_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]
        else:
            feature_names = column_names[:n_features]
            
            # If not enough names provided, extend with default names
            if len(feature_names) < n_features:
                feature_names.extend([f"feature_{i+len(feature_names)}" for i in range(n_features - len(feature_names))])
        
        # Convert to pandas if requested
        if as_pandas:
            X = pd.DataFrame(X, columns=feature_names)
            
        if return_feature_names:
            return X, feature_names
        return X

    def dataset_with_correlations(
        self,
        n_samples: int = 100,
        n_features: int = 5,
        correlation_strength: float = 0.8,
    ) -> tuple[np.ndarray, list[tuple[int, int, str]]]:
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
        X[:, 1] = correlation_strength * X[:, 0] + (
            1 - correlation_strength
        ) * self.rng.randn(n_samples)
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
        missing_pattern: str = "random",
    ) -> tuple[np.ndarray, np.ndarray]:
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
                replace=False,
            )
            for col in col_indices:
                row_indices = self.rng.choice(
                    n_samples,
                    size=max(1, int(n_samples * missing_rate)),
                    replace=False,
                )
                mask[row_indices, col] = True

        elif missing_pattern == "row":
            # Entire rows have missing values
            mask = np.zeros_like(X_complete, dtype=bool)
            row_indices = self.rng.choice(
                n_samples,
                size=max(1, int(n_samples * missing_rate)),
                replace=False,
            )
            for row in row_indices:
                col_indices = self.rng.choice(
                    n_features,
                    size=max(1, int(n_features * missing_rate)),
                    replace=False,
                )
                mask[row, col_indices] = True

        elif missing_pattern == "block":
            # Missing values form contiguous blocks
            mask = np.zeros_like(X_complete, dtype=bool)
            block_rows = max(1, int(n_samples * np.sqrt(missing_rate)))
            block_cols = max(1, int(n_features * np.sqrt(missing_rate)))

            row_start = self.rng.randint(0, n_samples - block_rows + 1)
            col_start = self.rng.randint(0, n_features - block_cols + 1)

            mask[
                row_start : row_start + block_rows,
                col_start : col_start + block_cols,
            ] = True

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
        outlier_type: str = "extreme",
    ) -> tuple[np.ndarray, list[int]]:
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
        categorical_levels: list[int] = None,
    ) -> tuple[np.ndarray, dict[str, list[int]]]:
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
            "binary": [],
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
            X[:, cat_idx] = np.digitize(
                X[:, cont_idx],
                bins=np.linspace(0, 1, levels + 1)[:-1],
            )

            # Add some noise to make it imperfect
            noise_mask = self.rng.rand(n_samples) < 0.2
            X[noise_mask, cat_idx] = self.rng.randint(0, levels, noise_mask.sum())

        return X, feature_types

    def dataset_with_special_values(
        self,
        n_samples: int = 100,
        n_features: int = 5,
    ) -> tuple[np.ndarray, dict[str, list[tuple[int, int]]]]:
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
        special_locations["negative"] = [
            (i, j) for i, j in zip(*np.where(negative_mask))
        ]

        return X, special_locations


    def generate_classification_data(
        self,
        n_samples: int = 100,
        n_features: int = 5,
        n_classes: int = 2,
        as_pandas: bool = False,
        column_names: list[str] = None,
        return_feature_names: bool = False,
        data_type: str = "basic"
    ):
        """Generate a synthetic classification dataset.

        Args:
            n_samples: Number of samples to generate
            n_features: Number of features to generate
            n_classes: Number of classes for the target variable
            as_pandas: Whether to return pandas DataFrame/Series instead of numpy arrays
            column_names: Custom column names for pandas DataFrame output
            return_feature_names: Whether to return feature names as a third element
            data_type: Type of dataset to generate:
                - "basic": Random values
                - "correlated": Features with correlations
                - "missing": Features with missing values
                - "outliers": Features with outliers
                - "mixed_types": Mixed numerical and categorical features
                - "special_values": Data with special numeric values (zero, large, etc.)

        Returns:
            X: Features (numpy array or pandas DataFrame)
            y: Target (numpy array or pandas Series)
            feature_names: (Optional) List of feature names if return_feature_names is True
            metadata: (Optional) Additional metadata about the dataset
        """
        # Generate features based on data_type
        feature_names = None
        metadata = None
        
        if data_type == "basic":
            if return_feature_names:
                X, feature_names = self.basic_dataset(
                    n_samples=n_samples,
                    n_features=n_features,
                    as_pandas=False,
                    column_names=column_names,
                    return_feature_names=True
                )
            else:
                X = self.basic_dataset(
                    n_samples=n_samples,
                    n_features=n_features,
                    as_pandas=False,
                    column_names=column_names
                )
        
        elif data_type == "correlated":
            X, correlations = self.dataset_with_correlations(
                n_samples=n_samples,
                n_features=n_features
            )
            metadata = {"correlations": correlations}
            
        elif data_type == "missing":
            X_complete, X = self.dataset_with_missing_values(
                n_samples=n_samples,
                n_features=n_features
            )
            metadata = {"complete_data": X_complete}
            
        elif data_type == "outliers":
            X, outlier_indices = self.dataset_with_outliers(
                n_samples=n_samples,
                n_features=n_features
            )
            metadata = {"outlier_indices": outlier_indices}
            
        elif data_type == "mixed_types":
            if n_features < 6:
                n_features = 6  # Ensure enough features for mixed types
            
            X, feature_types = self.mixed_type_dataset(
                n_samples=n_samples,
                n_continuous=n_features-3,
                n_categorical=2,
                n_binary=1
            )
            metadata = {"feature_types": feature_types}
            
        elif data_type == "special_values":
            X, special_locations = self.dataset_with_special_values(
                n_samples=n_samples,
                n_features=n_features
            )
            metadata = {"special_locations": special_locations}
            
        else:
            raise ValueError(f"Unknown data_type: {data_type}")
            
        # Generate class labels
        if n_classes == 2:
            # Binary classification - use the first feature to determine class
            y = (X[:, 0] > 0.5).astype(int)
        else:
            # Multi-class classification
            # Create some structure in the data so classes are somewhat meaningful
            # Use the first two features to determine class
            if n_features >= 2:
                # Create decision boundaries based on the first two features
                centroids = self.rng.rand(n_classes, 2)
                distances = np.zeros((n_samples, n_classes))
                
                for i in range(n_classes):
                    distances[:, i] = np.sqrt(
                        (X[:, 0] - centroids[i, 0])**2 + 
                        (X[:, 1] - centroids[i, 1])**2
                    )
                
                y = np.argmin(distances, axis=1)
            else:
                # Random assignments if not enough features
                y = self.rng.randint(0, n_classes, size=n_samples)
        
        # Convert to pandas if requested
        if as_pandas:
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(n_features)]
            X = pd.DataFrame(X, columns=feature_names)
            y = pd.Series(y, name="target")
        
        # Return with appropriate extras
        if return_feature_names and metadata:
            return X, y, feature_names, metadata
        elif return_feature_names:
            return X, y, feature_names
        elif metadata:
            return X, y, metadata
        return X, y

    def generate_regression_data(
        self,
        n_samples: int = 100,
        n_features: int = 5,
        noise: float = 0.1,
        as_pandas: bool = False,
        column_names: list[str] = None,
        return_feature_names: bool = False,
        data_type: str = "basic"
    ):
        """Generate a synthetic regression dataset.

        Args:
            n_samples: Number of samples to generate
            n_features: Number of features to generate
            noise: Amount of noise to add to the target
            as_pandas: Whether to return pandas DataFrame/Series instead of numpy arrays
            column_names: Custom column names for pandas DataFrame output
            return_feature_names: Whether to return feature names as a third element
            data_type: Type of dataset to generate:
                - "basic": Random values
                - "correlated": Features with correlations
                - "missing": Features with missing values
                - "outliers": Features with outliers
                - "mixed_types": Mixed numerical and categorical features
                - "special_values": Data with special numeric values (zero, large, etc.)

        Returns:
            X: Features (numpy array or pandas DataFrame)
            y: Target (numpy array or pandas Series)
            feature_names: (Optional) List of feature names if return_feature_names is True
            metadata: (Optional) Additional metadata about the dataset
        """
        # Generate features based on data_type
        feature_names = None
        metadata = None
        
        if data_type == "basic":
            if return_feature_names:
                X, feature_names = self.basic_dataset(
                    n_samples=n_samples,
                    n_features=n_features,
                    as_pandas=False,
                    column_names=column_names,
                    return_feature_names=True
                )
            else:
                X = self.basic_dataset(
                    n_samples=n_samples,
                    n_features=n_features,
                    as_pandas=False,
                    column_names=column_names
                )
        
        elif data_type == "correlated":
            X, correlations = self.dataset_with_correlations(
                n_samples=n_samples,
                n_features=n_features
            )
            metadata = {"correlations": correlations}
            
        elif data_type == "missing":
            X_complete, X = self.dataset_with_missing_values(
                n_samples=n_samples,
                n_features=n_features
            )
            metadata = {"complete_data": X_complete}
            
        elif data_type == "outliers":
            X, outlier_indices = self.dataset_with_outliers(
                n_samples=n_samples,
                n_features=n_features
            )
            metadata = {"outlier_indices": outlier_indices}
            
        elif data_type == "mixed_types":
            if n_features < 6:
                n_features = 6  # Ensure enough features for mixed types
            
            X, feature_types = self.mixed_type_dataset(
                n_samples=n_samples,
                n_continuous=n_features-3,
                n_categorical=2,
                n_binary=1
            )
            metadata = {"feature_types": feature_types}
            
        elif data_type == "special_values":
            X, special_locations = self.dataset_with_special_values(
                n_samples=n_samples,
                n_features=n_features
            )
            metadata = {"special_locations": special_locations}
            
        else:
            raise ValueError(f"Unknown data_type: {data_type}")
        
        # Generate coefficients for regression target
        if n_features >= 4:
            # Use a non-linear combination of features
            y = (
                2 * X[:, 0] + 
                1.5 * X[:, 1]**2 - 
                0.5 * X[:, 2] * X[:, 3] + 
                self.rng.normal(0, noise, size=n_samples)
            )
        else:
            # Simple linear relationship
            coefficients = self.rng.randn(n_features)
            y = X.dot(coefficients) + self.rng.normal(0, noise, size=n_samples)
        
        # Convert to pandas if requested
        if as_pandas:
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(n_features)]
            X = pd.DataFrame(X, columns=feature_names)
            y = pd.Series(y, name="target")
        
        # Return with appropriate extras
        if return_feature_names and metadata:
            return X, y, feature_names, metadata
        elif return_feature_names:
            return X, y, feature_names
        elif metadata:
            return X, y, metadata
        return X, y

    def dataset_with_mixed_types(
        self,
        n_samples: int = 100,
        n_numerical: int = 3,
        n_categorical: int = 2,
        max_categories: int = 5,
        as_pandas: bool = True
    ):
        """Generate a dataset with mixed numerical and categorical features.

        Args:
            n_samples: Number of samples to generate
            n_numerical: Number of numerical features
            n_categorical: Number of categorical features
            max_categories: Maximum number of categories per categorical feature
            as_pandas: Whether to return pandas DataFrame (always True for this method)

        Returns:
            DataFrame with mixed numerical and categorical features
        """
        # Generate numerical features
        X_num = self.rng.rand(n_samples, n_numerical)
        
        # Generate categorical features
        X_cat = np.zeros((n_samples, n_categorical), dtype=object)
        
        for i in range(n_categorical):
            n_categories = self.rng.randint(2, max_categories + 1)
            category_values = [f"cat_{i}_{j}" for j in range(n_categories)]
            X_cat[:, i] = self.rng.choice(category_values, size=n_samples)
        
        # Create column names
        num_cols = [f"num_{i}" for i in range(n_numerical)]
        cat_cols = [f"cat_{i}" for i in range(n_categorical)]
        
        # Combine into a DataFrame
        df_num = pd.DataFrame(X_num, columns=num_cols)
        df_cat = pd.DataFrame(X_cat, columns=cat_cols)
        
        # Return combined DataFrame
        return pd.concat([df_num, df_cat], axis=1)


@pytest.fixture
def dataset_generator():
    """Create a dataset generator with a fixed seed."""
    return DatasetGenerator(seed=42)

