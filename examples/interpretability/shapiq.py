"""This example demonstrates how to use the SHAP-IQ library to explain a TabPFN model."""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tabpfn_extensions import interpretability
from tabpfn import TabPFNClassifier

# Load example dataset
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names

# Set the number of model evaluations for the explanation, Note this number is very low for
# demonstration and testing purposes and should be increased for a real use cases
n_model_evals = 200

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=200, random_state=42
)
x_explain = X_test[0]

# Initialize and train model
clf = TabPFNClassifier()
clf.fit(X_train, y_train)

# Get a Shapley Interaction Explainer (here we use the Faithful Shapley Interaction Index)
explainer = interpretability.shapiq.get_tabpfn_explainer(
    model=clf,
    data=X_train,
    labels=y_train,
    index="FSII",    # SV: Shapley Value, FSII: Faithful Shapley Interaction Index
    max_order=2,     # maximum order of the Shapley interactions (2 for pairwise interactions)
    verbose=True,    # show a progress bar during explanation
)
shapley_interaction_values = explainer.explain(x=x_explain, budget=n_model_evals)
shapley_interaction_values.plot_upset(feature_names=feature_names)  # plot an upset plot

# Initialize and train model (we need to contextualize the model with the whole training data again)
clf = TabPFNClassifier()
clf.fit(X_train, y_train)

# Get a TabPFNExplainer and explain the model
explainer = interpretability.shapiq.get_tabpfn_imputation_explainer(
    model=clf,
    data=X_test[:50],   # use a subset of the test data for the background data
    imputer="marginal",  # use marginal imputation
    index="SV",          # SV: Shapley Value (like in shap)
    verbose=True,        # show a progress bar during explanation
)

shapley_values = explainer.explain(x=x_explain, budget=n_model_evals)
shapley_values.plot_force(feature_names=feature_names)  # plot the force plot
