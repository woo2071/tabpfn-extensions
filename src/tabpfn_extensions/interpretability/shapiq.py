#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0
from typing import Optional, Union, Literal

import numpy as np
import pandas as pd
import tabpfn


def get_tabpfn_explainer(
    model: Union[tabpfn.TabPFNRegressor, tabpfn.TabPFNClassifier],
    data: Union[pd.DataFrame, np.ndarray],
    labels: Union[pd.DataFrame, np.ndarray],
    index: Union[Literal["SV"], Literal["k-SII"], Literal["FSII"], Literal["STII"], Literal["SII"]] = "k-SII",
    max_order: int = 2,
    class_index: Optional[int] = None,
    **kwargs
):
    """Get a TabPFNExplainer from shapiq.

    This function returns the TabPFN explainer from the shapiq[1]_ library. The explainer uses
    a remove-and-recontextualize paradigm of model explanation[2]_[3]_ to explain the predictions
    of a TabPFN model. See ``shapiq.TabPFNExplainer`` documentation for more information regarding
    the explainer object.

    Args:
        model (tabpfn.TabPFNRegressor or tabpfn.TabPFNClassifier): The TabPFN model to explain.

        data (pd.DataFrame or np.ndarray): The background data to use for the explainer.

        labels (pd.DataFrame or np.ndarray): The labels for the background data.

        index: The index to use for the explanation. Options are:
            - ``"SV"``: Shapley values, similar to SHAP.
            - ``SII``: Shapley interaction index (not efficient, i.e. values do not add up to the
                model prediciton).
            - ``FSII``: Faithful Shapley interaction index.
            - ``k-SII``: k-Shapley interaction index.
            - ``STII``: Shapley-Taylor interaction index.

        max_order (int): The maximum order of interactions to consider. Defaults to 2.

        class_index (int, optional): The class index of the model to explain. If not provided, the
            class index will be set to 1 per default for classification models. This argument is
            ignored for regression models. Defaults to None.

        **kwargs: Additional keyword arguments to pass to the explainer.

    Returns:
        shapiq.TabPFNExplainer: The TabPFN explainer.

    References:
        .. [1] shapiq repository: https://github.com/mmschlk/shapiq
        .. [2] Muschalik, M., Baniecki, H., Fumagalli, F., Kolpaczki, P., Hammer, B., Hüllermeier, E. (2024). shapiq: Shapley Interactions for Machine Learning. In: The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track. url: https://openreview.net/forum?id=knxGmi6SJi
        .. [3] Rundel, D., Kobialka, J., von Crailsheim, C., Feurer, M., Nagler, T., Rügamer, D. (2024). Interpretable Machine Learning for TabPFN. In: Longo, L., Lapuschkin, S., Seifert, C. (eds) Explainable Artificial Intelligence. xAI 2024. Communications in Computer and Information Science, vol 2154. Springer, Cham. https://doi.org/10.1007/978-3-031-63797-1_23

    """
    import shapiq

    # make data to array if it is a pandas DataFrame
    if isinstance(data, pd.DataFrame):
        data = data.values

    # make labels to array if it is a pandas Series
    if isinstance(labels, pd.Series) or isinstance(labels, pd.DataFrame):
        labels = labels.values

    return shapiq.TabPFNExplainer(
        model=model,
        data=data,
        labels=labels,
        index=index,
        max_order=max_order,
        class_index=class_index,
        **kwargs,
    )
