#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor
from tabpfn_extensions.rf_pfn import (
    RandomForestTabPFNClassifier,
    RandomForestTabPFNRegressor,
)

clf_base = TabPFNClassifier()
reg_base = TabPFNRegressor()

# Check if this is being run in a test environment
X, y = load_breast_cancer(return_X_y=True)
test_size = 0.33
n_estimators = 10  # Default value

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=42,
)

# Create and fit classifier with appropriate settings
clf = RandomForestTabPFNClassifier(
    tabpfn=clf_base,
    n_estimators=n_estimators,
    max_depth=3,  # Use shallow trees for faster training
)
clf.fit(X_train, y_train)
prediction_probabilities = clf.predict_proba(X_test)
predictions = np.argmax(prediction_probabilities, axis=-1)

print("ROC AUC:", roc_auc_score(y_test, prediction_probabilities[:, 1]))
print("Accuracy", accuracy_score(y_test, predictions))

# Check for test environment by looking for the FAST_TEST_MODE environment variable
import os

is_test = os.environ.get("FAST_TEST_MODE", "0") == "1"

# Multiclass - Use smaller dataset in test mode
if is_test:
    # Use minimal synthetic dataset for faster testing
    np.random.seed(43)
    X = np.random.rand(15, 3)
    y = np.random.randint(0, 3, size=15)
    test_size = 0.4
    n_estimators = 1
else:
    # Normal operation - use real dataset
    X, y = load_iris(return_X_y=True)
    test_size = 0.33
    n_estimators = 10

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=42,
)

# Create classifier with appropriate settings for test/normal mode
clf = RandomForestTabPFNClassifier(
    tabpfn=clf_base,
    n_estimators=n_estimators,
    max_depth=2 if is_test else 5,
)
clf.fit(X_train, y_train)
prediction_probabilities = clf.predict_proba(X_test)
predictions = np.argmax(prediction_probabilities, axis=-1)

print("Accuracy", accuracy_score(y_test, predictions))
# Only calculate ROC AUC if we have enough samples per class
if not is_test or len(np.unique(y_test)) <= 2:
    print(
        "ROC AUC:",
        roc_auc_score(y_test, prediction_probabilities, multi_class="ovr"),
    )

# Regression
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.33,
    random_state=42,
)
reg = RandomForestTabPFNRegressor(tabpfn=reg_base)
reg.fit(X_train, y_train)
predictions = reg.predict(X_test)
print("Mean Squared Error (MSE):", mean_squared_error(y_test, predictions))
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, predictions))
print("R-squared (R^2):", r2_score(y_test, predictions))
