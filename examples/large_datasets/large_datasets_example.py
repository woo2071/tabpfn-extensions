# ----------------------------
# Install and Import Libraries
# ----------------------------
import numpy as np  # Numerical computations
from sklearn.datasets import fetch_openml, load_diabetes  # Datasets
from sklearn.metrics import (  # Evaluation metrics
    accuracy_score,  # Classification accuracy
    mean_absolute_error,  # Regression MAE
    mean_squared_error,  # Regression MSE
    r2_score,  # Regression R^2 score
    roc_auc_score,  # Area Under ROC Curve
)
from sklearn.model_selection import train_test_split  # Train-test splitting
from sklearn.preprocessing import LabelEncoder  # Encoding categorical variables

from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor  # TabPFN models
from tabpfn_extensions.rf_pfn import (  # RF extensions
    RandomForestTabPFNClassifier,
    RandomForestTabPFNRegressor,
)

# ----------------------------
# Load and Preprocess Classification Dataset
# ----------------------------
print("\n--- Loading Classification Dataset ---")
df_classification = fetch_openml(
    "electricity",  # Dataset name from OpenML
    version=1,  # Specific dataset version
    as_frame=True,  # Return data as pandas DataFrame
)
X_class, y_class = (
    df_classification.data,
    df_classification.target,
)  # Feature matrix and target vector

le = LabelEncoder()  # Initialize label encoder
y_class = le.fit_transform(y_class)  # Encode target variable into numeric classes

# Convert all categorical columns to numeric codes
for col in X_class.select_dtypes(["category"]).columns:
    X_class[col] = X_class[
        col
    ].cat.codes  # Convert categorical features to numeric codes

# Split data into training and testing sets
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
    X_class,  # Feature matrix
    y_class,  # Target vector
    test_size=0.33,  # Fraction of data used for testing
    random_state=42,  # Ensures reproducible splits
)

# ----------------------------
# Classification - Strategy 1: Subsampled Ensemble using TabPFNClassifier
# ----------------------------
print("\n--- Classification: Strategy 1 (Subsampled Ensemble) ---")
tabpfn_subsample_clf = TabPFNClassifier(
    ignore_pretraining_limits=True,  # Allow larger datasets beyond pretraining limits
    n_estimators=32,  # Number of ensemble estimators (more improves performance but increases runtime)
    inference_config={
        "SUBSAMPLE_SAMPLES": 10000,  # Max samples per inference to manage memory usage
    },
)

tabpfn_subsample_clf.fit(X_train_class, y_train_class)  # Train the classifier
prediction_probabilities = tabpfn_subsample_clf.predict_proba(
    X_test_class
)  # Probability predictions
predictions = np.argmax(
    prediction_probabilities, axis=1
)  # Convert probabilities to class predictions

# Evaluate classification performance
print(f"ROC AUC: {roc_auc_score(y_test_class, prediction_probabilities[:, 1]):.4f}")
print(f"Accuracy: {accuracy_score(y_test_class, predictions):.4f}")

# ----------------------------
# Classification - Strategy 2: Tree-based Model with Node Subsampling
# ----------------------------
print("\n--- Classification: Strategy 2 (Tree-based Model) ---")
clf_base = TabPFNClassifier(
    ignore_pretraining_limits=True,  # Enable larger datasets
    inference_config={"SUBSAMPLE_SAMPLES": 10000},  # Control memory usage
)

tabpfn_tree_clf = RandomForestTabPFNClassifier(
    tabpfn=clf_base,  # Base TabPFN model used within the Random Forest
    verbose=1,  # Print detailed training progress
    max_predict_time=60,  # Max time (in seconds) for making predictions
)

tabpfn_tree_clf.fit(X_train_class, y_train_class)  # Train the tree-based classifier
prediction_probabilities = tabpfn_tree_clf.predict_proba(
    X_test_class
)  # Probability predictions
predictions = np.argmax(prediction_probabilities, axis=1)  # Class predictions

# Evaluate classification performance
print(f"ROC AUC: {roc_auc_score(y_test_class, prediction_probabilities[:, 1]):.4f}")
print(f"Accuracy: {accuracy_score(y_test_class, predictions):.4f}")

# ----------------------------
# Load and Preprocess Regression Dataset
# ----------------------------
print("\n--- Loading Regression Dataset ---")
regression_data = load_diabetes()  # Load diabetes dataset for regression
X_reg, y_reg = (
    regression_data.data,
    regression_data.target,
)  # Feature matrix and target vector

# Split data into training and testing sets
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg,  # Feature matrix
    y_reg,  # Target vector
    test_size=0.33,  # Fraction for testing
    random_state=42,  # Reproducibility
)

# ----------------------------
# Regression - Strategy 1: Subsampled Ensemble using TabPFNRegressor
# ----------------------------
print("\n--- Regression: Strategy 1 (Subsampled Ensemble) ---")
tabpfn_subsample_reg = TabPFNRegressor(
    ignore_pretraining_limits=True,  # Allow handling of larger datasets
    n_estimators=32,  # Number of estimators in ensemble
    inference_config={
        "SUBSAMPLE_SAMPLES": 10000,  # Limit samples per inference to prevent OOM errors
    },
)

tabpfn_subsample_reg.fit(X_train_reg, y_train_reg)  # Train the regressor
reg_predictions = tabpfn_subsample_reg.predict(X_test_reg)  # Make predictions

# Evaluate regression performance
print(
    f"Mean Squared Error (MSE): {mean_squared_error(y_test_reg, reg_predictions):.4f}"
)
print(
    f"Mean Absolute Error (MAE): {mean_absolute_error(y_test_reg, reg_predictions):.4f}"
)
print(f"R^2 Score: {r2_score(y_test_reg, reg_predictions):.4f}")

# ----------------------------
# Regression - Strategy 2: Tree-based Model with Node Subsampling
# ----------------------------
print("\n--- Regression: Strategy 2 (Tree-based Model) ---")
reg_base = TabPFNRegressor(
    ignore_pretraining_limits=True,  # Enable large dataset support
    inference_config={"SUBSAMPLE_SAMPLES": 10000},  # Memory management
)

tabpfn_tree_reg = RandomForestTabPFNRegressor(
    tabpfn=reg_base,  # Base TabPFN regressor
    verbose=1,  # Show detailed logs
    max_predict_time=60,  # Max prediction time in seconds
)

tabpfn_tree_reg.fit(X_train_reg, y_train_reg)  # Train the model
reg_tree_predictions = tabpfn_tree_reg.predict(X_test_reg)  # Predict on test data

# Evaluate regression performance
print(
    f"Mean Squared Error (MSE): {mean_squared_error(y_test_reg, reg_tree_predictions):.4f}"
)
print(
    f"Mean Absolute Error (MAE): {mean_absolute_error(y_test_reg, reg_tree_predictions):.4f}"
)
print(f"R^2 Score: {r2_score(y_test_reg, reg_tree_predictions):.4f}")

# ----------------------------
# Settings Overview:
# ----------------------------
# - n_estimators: Controls ensemble size; higher improves accuracy but increases runtime.
# - SUBSAMPLE_SAMPLES: Prevents out-of-memory errors by limiting samples per inference.
# - max_predict_time: Limits prediction time to manage computational budgets.
# - random_state: Ensures reproducible train-test splits.

# ----------------------------
# Notes:
# ----------------------------
# - Classification uses the 'electricity' dataset; regression uses the diabetes dataset.
# - Ensure that the `tabpfn_extensions` package is installed with optional dependencies.
# - Increasing `n_estimators` enhances model stability at the cost of longer training times.
# - Subsampling is essential for large datasets to avoid memory issues.
# - Verbose mode helps monitor training progress for debugging or long runs.
