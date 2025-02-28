# ClassifierAsRegressor

This extension enables using TabPFN's powerful classifier for regression tasks. It works by:

1. Discretizing the continuous target values into categorical bins
2. Training a classifier on these bins
3. Converting classifier predictions back to continuous values

## Basic Usage

```python
from tabpfn_extensions.classifier_as_regressor import ClassifierAsRegressor
from tabpfn import TabPFNClassifier  # or from tabpfn_client

# Create the base classifier
base_classifier = TabPFNClassifier()

# Wrap it with ClassifierAsRegressor
regressor = ClassifierAsRegressor(estimator=base_classifier)

# Use like any scikit-learn regressor
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

# Get probability distributions over target values
full_pred = regressor.predict_full(X_test)
mean = full_pred["mean"]         # Mean prediction
var = full_pred["variance"]      # Variance of prediction
probs = full_pred["buckets"]     # Probabilities for each bin
```

## Features

- **Seamless Integration**: Works with any classifier that has a `predict_proba` method
- **Distribution Information**: Returns full probability distributions over target values
- **Backend Compatibility**: Works with both TabPFN and TabPFN-client
- **scikit-learn Compatible**: Follows scikit-learn regressor API

## Tips

- For optimal performance, use a classifier that produces well-calibrated probabilities
- The granularity of regression can be controlled by the number of unique classes in the training data
- Works especially well for problems where the target has natural "bins" or clusters

## Reference

For more detailed information, see the [example](../../../../examples/classifier_as_regressor/) or [source code](../classifier_as_regressor.py).