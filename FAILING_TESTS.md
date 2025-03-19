# Failing Tests in TabPFN Extensions

## Detailed List of Failing Tests

1. **Decision Tree Tests**
   - `tests/test_dt_pfn.py::TestDecisionTreeRegressor::test_tree_model_behavior[backend=tabpfn]`
   - `tests/test_dt_pfn.py::TestDecisionTreeRegressor::test_tree_model_behavior[backend=tabpfn_client]`

2. **HPO Tests**
   - `tests/test_hpo.py::TestTunedTabPFNClassifier::test_fit_predict[backend=tabpfn]`
   - `tests/test_hpo.py::TestTunedTabPFNClassifier::test_fit_predict[backend=tabpfn_client]`
   - `tests/test_hpo.py::TestTunedTabPFNClassifier::test_predict_proba[backend=tabpfn]`
   - `tests/test_hpo.py::TestTunedTabPFNClassifier::test_predict_proba[backend=tabpfn_client]`
   - `tests/test_hpo.py::TestTunedTabPFNClassifier::test_with_pandas[backend=tabpfn]`
   - `tests/test_hpo.py::TestTunedTabPFNClassifier::test_with_pandas[backend=tabpfn_client]`
   - `tests/test_hpo.py::TestTunedTabPFNClassifier::test_with_multiclass[backend=tabpfn]`
   - `tests/test_hpo.py::TestTunedTabPFNClassifier::test_with_multiclass[backend=tabpfn_client]`
   - `tests/test_hpo.py::TestTunedTabPFNClassifier::test_with_missing_values[backend=tabpfn]`
   - `tests/test_hpo.py::TestTunedTabPFNClassifier::test_with_missing_values[backend=tabpfn_client]`
   - `tests/test_hpo.py::TestTunedTabPFNClassifier::test_with_text_features[backend=tabpfn]`
   - `tests/test_hpo.py::TestTunedTabPFNClassifier::test_with_text_features[backend=tabpfn_client]`
   - `tests/test_hpo.py::TestTunedTabPFNClassifier::test_extreme_cases[backend=tabpfn]`
   - `tests/test_hpo.py::TestTunedTabPFNClassifier::test_extreme_cases[backend=tabpfn_client]`
   - `tests/test_hpo.py::TestTunedTabPFNRegressor::test_fit_predict[backend=tabpfn]`
   - `tests/test_hpo.py::TestTunedTabPFNRegressor::test_fit_predict[backend=tabpfn_client]`
   - `tests/test_hpo.py::TestTunedTabPFNRegressor::test_with_pandas[backend=tabpfn]`
   - `tests/test_hpo.py::TestTunedTabPFNRegressor::test_with_pandas[backend=tabpfn_client]`
   - `tests/test_hpo.py::TestTunedTabPFNRegressor::test_with_missing_values[backend=tabpfn]`
   - `tests/test_hpo.py::TestTunedTabPFNRegressor::test_with_missing_values[backend=tabpfn_client]`
   - `tests/test_hpo.py::TestTunedTabPFNRegressor::test_with_text_features[backend=tabpfn]`
   - `tests/test_hpo.py::TestTunedTabPFNRegressor::test_with_text_features[backend=tabpfn_client]`
   - `tests/test_hpo.py::TestTunedTabPFNRegressor::test_extreme_cases[backend=tabpfn]`
   - `tests/test_hpo.py::TestTunedTabPFNRegressor::test_extreme_cases[backend=tabpfn_client]`
   - `tests/test_hpo.py::TestHPOSpecificFeatures::test_different_metrics`

3. **Post-Hoc Ensembles Tests**
   - `tests/test_post_hoc_ensembles.py::TestAutoTabPFNClassifier::test_extreme_cases[backend=tabpfn]`
   - `tests/test_post_hoc_ensembles.py::TestAutoTabPFNClassifier::test_extreme_cases[backend=tabpfn_client]`

## Common Error Patterns

1. **Decision Tree Issues:**
   - `assert not True` in tree model behavior tests
   - Indicates that arrays are equal when they shouldn't be

2. **HPO Issues:**
   - `AssertionError: Poly. features to add must be >0!`
   - Problem with polynomial feature settings in preprocessing configuration

3. **Post-Hoc Ensembles Issues:**
   - `ValueError: Only one class present in y_true. ROC AUC score is not defined in that case.`
   - Problem with metrics in extreme case tests

## Performance Notes

Slowest Tests:
1. `tests/test_dt_pfn.py::TestDecisionTreeRegressor::test_with_missing_values[backend=tabpfn_client]` - 16.33s
2. `tests/test_dt_pfn.py::TestDecisionTreeRegressor::test_tree_model_behavior[backend=tabpfn_client]` - 13.84s
3. `tests/test_dt_pfn.py::TestDecisionTreeRegressor::test_extreme_cases[backend=tabpfn_client]` - 11.16s
4. `tests/test_dt_pfn.py::TestDecisionTreeRegressor::test_fit_predict[backend=tabpfn_client]` - 9.96s
5. `tests/test_post_hoc_ensembles.py::TestAutoTabPFNRegressor::test_extreme_cases[backend=tabpfn]` - 9.82s
6. `tests/test_post_hoc_ensembles.py::TestAutoTabPFNRegressor::test_extreme_cases[backend=tabpfn_client]` - 9.67s

## Priority Order for Fixing

1. **HPO polynomial features issue**
   - Fix the `Poly. features to add must be >0!` error by updating search space configuration
   - This error affects all HPO tests

2. **Decision Tree validation issue**
   - Fix the tree model behavior test's equality assertion
   - Update test expectations to account for valid numerical differences between implementations

3. **Post-Hoc Ensembles metrics issue**
   - Fix ROC AUC score issues in extreme case tests
   - Adjust test case to ensure multiple classes are present
