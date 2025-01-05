#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

import os


def is_tabpfn(estimator):
    if "TabPFN" in str(estimator.__class__):
        return True
    else:
        return False
    if issubclass(estimator.__class__, TabPFNBaseModel):
        return True
    else:
        return False

USE_TABPFN_LOCAL = os.getenv("USE_TABPFN_LOCAL", "true")

if USE_TABPFN_LOCAL.lower() == "true":
    from tabpfn import TabPFNClassifier, TabPFNRegressor
else:
    from tabpfn_client import TabPFNClassifier, TabPFNRegressor