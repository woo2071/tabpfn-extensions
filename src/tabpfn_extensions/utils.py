#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

from typing import Union, TypeVar


def is_tabpfn(estimator):
    if "TabPFN" in str(estimator.__class__):
        return True
    else:
        return False
    if issubclass(estimator.__class__, TabPFNBaseModel):
        return True
    else:
        return False


# Regressors
TabPFNClientRegressor = TypeVar(
    "TabPFNRegressor", bound="tabpfn_client.estimator.TabPFNRegressor"
)
TabPFNLocalRegressor = TypeVar(
    "TabPFNRegressor", bound="tabpfn.estimator.TabPFNRegressor"
)

# Classifiers
TabPFNClientClassifier = TypeVar(
    "TabPFNClassifier", bound="tabpfn_client.estimator.TabPFNClassifier"
)
TabPFNLocalClassifier = TypeVar(
    "TabPFNClassifier", bound="tabpfn.estimator.TabPFNClassifier"
)

# Define the union types for either model
TabPFNRegressor = Union[TabPFNClientRegressor, TabPFNLocalRegressor]
TabPFNClassifier = Union[TabPFNClientClassifier, TabPFNLocalClassifier]
