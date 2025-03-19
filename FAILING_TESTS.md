# Failing Tests in TabPFN Extensions

## Detailed List of Failing Tests

1. **Post-Hoc Ensembles Tests**
   - `tests/test_post_hoc_ensembles.py::TestAutoTabPFNClassifier::test_fit_predict`
   - `tests/test_post_hoc_ensembles.py::TestAutoTabPFNClassifier::test_predict_proba`
   - `tests/test_post_hoc_ensembles.py::TestAutoTabPFNClassifier::test_with_pandas`
   - `tests/test_post_hoc_ensembles.py::TestAutoTabPFNClassifier::test_with_multiclass`
   - `tests/test_post_hoc_ensembles.py::TestAutoTabPFNClassifier::test_with_missing_values`
   - `tests/test_post_hoc_ensembles.py::TestAutoTabPFNClassifier::test_with_text_features`
   - `tests/test_post_hoc_ensembles.py::TestAutoTabPFNClassifier::test_extreme_cases`
   - `tests/test_post_hoc_ensembles.py::TestAutoTabPFNClassifier::test_passes_estimator_checks`
   - `tests/test_post_hoc_ensembles.py::TestAutoTabPFNRegressor::test_fit_predict`
   - `tests/test_post_hoc_ensembles.py::TestAutoTabPFNRegressor::test_with_pandas`
   - `tests/test_post_hoc_ensembles.py::TestAutoTabPFNRegressor::test_with_missing_values`
   - `tests/test_post_hoc_ensembles.py::TestAutoTabPFNRegressor::test_with_text_features`
   - `tests/test_post_hoc_ensembles.py::TestAutoTabPFNRegressor::test_extreme_cases`
   - `tests/test_post_hoc_ensembles.py::TestAutoTabPFNRegressor::test_passes_estimator_checks`

2. **HPO Tests**
   - `tests/test_hpo.py::TestTunedTabPFNClassifier::test_with_multiclass`
   - `tests/test_hpo.py::TestTunedTabPFNClassifier::test_with_text_features`
   - `tests/test_hpo.py::TestTunedTabPFNClassifier::test_passes_estimator_checks`
   - `tests/test_hpo.py::TestTunedTabPFNRegressor::test_fit_predict`
   - `tests/test_hpo.py::TestTunedTabPFNRegressor::test_with_pandas`
   - `tests/test_hpo.py::TestTunedTabPFNRegressor::test_with_missing_values`
   - `tests/test_hpo.py::TestTunedTabPFNRegressor::test_with_text_features`
   - `tests/test_hpo.py::TestTunedTabPFNRegressor::test_extreme_cases`
   - `tests/test_hpo.py::TestTunedTabPFNRegressor::test_passes_estimator_checks`
   - `tests/test_hpo.py::TestHPOSpecificFeatures::test_different_metrics`

3. **Decision Tree Tests**
   - `tests/test_dt_pfn.py::TestDecisionTreeClassifier::test_passes_estimator_checks`
   - `tests/test_dt_pfn.py::TestDecisionTreeRegressor::test_passes_estimator_checks`

4. **Embedding Tests**
   - `tests/test_embedding.py::TestTabPFNEmbedding::test_clf_embedding_vanilla`
   - `tests/test_embedding.py::TestTabPFNEmbedding::test_clf_embedding_kfold`
   - `tests/test_embedding.py::TestTabPFNEmbedding::test_reg_embedding_vanilla`

5. **Example Tests**
   - `tests/test_examples.py::test_example[shapiq_example.py]`
   - `tests/test_examples.py::test_example[large_datasets_example.py]`
   - `tests/test_examples.py::test_example[get_embeddings.py]`

6. **Random Forest Tests**
   - `tests/test_rf_pfn.py::TestRandomForestClassifier::test_passes_estimator_checks`
   - `tests/test_rf_pfn.py::TestRandomForestRegressor::test_passes_estimator_checks`

## Common Error Patterns

1. **TabPFN Preprocessing Compatibility Issues:**
   - `AttributeError: 'dict' object has no attribute 'name'`
   - Error in TabPFN 2.0.6 where a dict is expected to have a 'name' attribute

2. **Embeddings Feature Not Implemented:**
   - `AttributeError: 'TabPFNClassifier' object has no attribute 'get_embeddings'`

3. **Test Timeouts:**
   - Some tests time out, especially with the tabpfn_client backend

4. **Pandas Index Errors:**
   - `KeyError: '[11, 2, 5, 9, 8] not in index'` in Random Forest tests

## Other Issues

- `test_examples.py` failures with specific examples
- Estimator attribute issues in the Decision Tree tests

## Priority Order for Fixing

1. Post-Hoc Ensembles TabPFN 2.0 compatibility issues (most critical blocking issue)
2. HPO compatibility issues (same root cause as #1)
3. Decision Tree estimator implementation issues
4. RF-PFN pandas handling issues
5. Embedding feature implementation
6. Example test issues
