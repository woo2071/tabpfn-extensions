# TabPFN Interpretability

## TabPFN shapiq

``shapiq`` is a library for computing Shapley-based explanations like Shapley values or Shapley
interactions for machine learning models. ``shapiq`` offers native support for interpreting TabPFN
by utilizing a remove-and-recontextualize paradigm of model interpretation. The library extends the
well-known SHAP library by providing a more efficient and scalable implementation of Shapley values
and Shapley interactions. The ``shapiq`` library can be cited as follows:

```bibtext
@inproceedings{muschalik2024shapiq,
  title     = {shapiq: Shapley Interactions for Machine Learning},
  author    = {Maximilian Muschalik and Hubert Baniecki and Fabian Fumagalli and
               Patrick Kolpaczki and Barbara Hammer and Eyke H\"{u}llermeier},
  booktitle = {The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year      = {2024},
  url       = {https://openreview.net/forum?id=knxGmi6SJi}
}
```

## TabPFN SHAP

``shap`` is a library for computing Shapley values explanations for machine learning models.
``shap`` constructs these explanations by imputing missing values with randomly drawn samples from
a background distribution. The ``shap`` library can be cited as follows:

```bibtext
@inproceedings{DBLP:conf/nips/LundbergL17,
  author       = {Scott M. Lundberg and
                  Su{-}In Lee},
  title        = {A Unified Approach to Interpreting Model Predictions},
  booktitle    = {Advances in Neural Information Processing Systems 30},
  pages        = {4765--4774},
  year         = {2017},
  url          = {https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html},
}
```
