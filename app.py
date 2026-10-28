from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore", category=UserWarning)
ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "trainset.xlsx"
TEST_PATH = ROOT / "test set.xlsx"

st.set_page_config(page_title="Concrete Strength AI Lab", page_icon="🧪", layout="wide")
st.title("Concrete Compressive Strength | Model Comparison Lab")
st.caption("Compare three regression models and a stacked ensemble using the supplied train/test split.")
st.info("Target: concrete compressive strength (MPa). All model scores below are calculated on the same held-out test set.")

@st.cache_data
def load_data():
    train = pd.read_excel(TRAIN_PATH)
    test = pd.read_excel(TEST_PATH)
    train.columns = [str(c).strip() for c in train.columns]
    test.columns = [str(c).strip() for c in test.columns]
    target = next((c for c in train.columns if "compressive strength" in c.lower()), train.columns[-1])
    features = [c for c in train.columns if c != target]
    train = train[features + [target]].dropna()
    test = test[features + [target]].dropna()
    return train, test, features, target

def build_models(features_count):
    # Degree 2 was selected from 5-fold CV among polynomial degrees 1, 2 and 3.
    poly = make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False),
        StandardScaler(),
        LinearRegression()
    )
    forest = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    # MLPRegressor is a scikit-learn feed-forward ANN with the same 128-64-32 hidden-layer pattern.
    neural = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            max_iter=1200,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=30,
            random_state=42
        )
    )
    return {
        "Polynomial Regression": poly,
        "Random Forest": forest,
        "Neural Network (ANN)": neural
    }

@st.cache_resource(show_spinner="Training the three models and fitting the combined model…")
def train_and_evaluate(train_csv_signature, test_csv_signature):
    train, test, features, target = load_data()
    X_train, y_train = train[features], train[target]
    X_test, y_test = test[features], test[target]
    models = build_models(len(features))
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Generate out-of-fold predictions for the meta-model. Each training row's
    # meta-feature is predicted by a base estimator that did not train on that row.
    oof_predictions = []
    for model in models.values():
        oof_predictions.append(cross_val_predict(clone(model), X_train, y_train, cv=cv, n_jobs=None))
    oof_matrix = np.column_stack(oof_predictions)

    # Ridge learns how to combine the three model outputs; it is not fitted on test labels.
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(oof_matrix, y_train)

    test_predictions = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        test_predictions[name] = model.predict(X_test)

    ensemble_prediction = meta_model.predict(np.column_stack(list(test_predictions.values())))
    test_predictions["Combined Stacking Model"] = ensemble_prediction

    rows = []
    for name, pred in test_predictions.items():
        rows.append({
            "Model": name,
            "R²": r2_score(y_test, pred),
            "MAE (MPa)": mean_absolute_error(y_test, pred),
            "RMSE (MPa)": np.sqrt(mean_squared_error(y_test, pred))
        })
    metrics = pd.DataFrame(rows)
    actual_pred = pd.DataFrame({"Actual strength (MPa)": y_test.to_numpy()})
    for name, pred in test_predictions.items():
        actual_pred[name + " prediction"] = pred
    weights = pd.DataFrame({
        "Base model": list(models.keys()),
        "Meta-model coefficient": meta_model.coef_
    })
    return metrics, test_predictions, actual_pred, models, meta_model, weights, features, target, X_test, y_test

try:
    train, test, features, target = load_data()
except Exception as exc:
    st.error(f"Could not load the included Excel datasets: {exc}")
    st.stop()

# Use file metadata in the cache key so changed datasets cause a fresh fit.
train_sig = f"{TRAIN_PATH.stat().st_size}-{TRAIN_PATH.stat().st_mtime_ns}"
test_sig = f"{TEST_PATH.stat().st_size}-{TEST_PATH.stat().st_mtime_ns}"
try:
    metrics, predictions, actual_pred, fitted_models, meta_model, weights, features, target, X_test, y_test = train_and_evaluate(train_sig, test_sig)
except Exception as exc:
    st.error(f"Model training failed: {exc}")
    st.exception(exc)
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Training rows", f"{len(train):,}")
c2.metric("Held-out test rows", f"{len(test):,}")
c3.metric("Input features", str(len(features)))

st.subheader("1. Test-set performance")
st.dataframe(
    metrics.style.format({"R²": "{:.4f}", "MAE (MPa)": "{:.3f}", "RMSE (MPa)": "{:.3f}"}),
    use_container_width=True,
    hide_index=True
)
best_base = metrics[metrics["Model"] != "Combined Stacking Model"]["R²"].max()
ensemble_r2 = float(metrics.loc[metrics["Model"] == "Combined Stacking Model", "R²"].iloc[0])
delta = ensemble_r2 - float(best_base)
st.metric("Combined model R² improvement over best individual model", f"{delta:+.4f}", help="Difference in R² on the same held-out test set. A positive value indicates improvement.")
if delta > 0:
    st.success(f"The combined model improves test-set R² by {delta:.4f} over the strongest individual model in this run.")
else:
    st.warning("The combined model did not outperform the strongest individual model on this test split. The displayed score is the measured result; it has not been altered.")

st.subheader("2. R² comparison")
fig, ax = plt.subplots(figsize=(10, 4.5))
plot_data = metrics.sort_values("R²", ascending=False)
bars = ax.bar(plot_data["Model"], plot_data["R²"])
ax.set_ylabel("R² score (higher is better)")
ax.set_ylim(min(-0.1, float(plot_data["R²"].min()) - 0.05), 1.0)
ax.tick_params(axis="x", rotation=15)
ax.grid(axis="y", alpha=0.25)
for bar, value in zip(bars, plot_data["R²"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.012, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
st.pyplot(fig, clear_figure=True)

st.subheader("3. Actual vs predicted")
selected_model = st.selectbox("Choose model to inspect", list(predictions.keys()), index=list(predictions.keys()).index("Combined Stacking Model"))
pred = predictions[selected_model]
fig2, ax2 = plt.subplots(figsize=(7, 5))
ax2.scatter(y_test, pred, alpha=0.7)
lo = min(float(np.min(y_test)), float(np.min(pred)))
hi = max(float(np.max(y_test)), float(np.max(pred)))
ax2.plot([lo, hi], [lo, hi], "--", label="Ideal prediction")
ax2.set_xlabel("Actual compressive strength (MPa)")
ax2.set_ylabel("Predicted compressive strength (MPa)")
ax2.set_title(selected_model)
ax2.legend()
ax2.grid(alpha=0.25)
st.pyplot(fig2, clear_figure=True)

st.subheader("4. How the combined model works")
st.markdown("""
The combined model is a **stacking ensemble**. It does not simply average the predictions:
1. **Polynomial Regression** learns a smooth relationship and degree-2 feature interactions.
2. **Random Forest** averages many decision trees to learn nonlinear patterns and feature thresholds.
3. **Neural Network (ANN)** learns nonlinear relationships through layers of connected neurons.
4. A **Ridge Regression meta-model** learns how to combine the three model predictions into one final estimate.

To reduce data leakage, the meta-model is trained on **5-fold out-of-fold predictions** from the training set. Each row's training-time base-model predictions come from a model that did not train on that row. The test set is used only for final evaluation.
""")
st.markdown("**Learned combination coefficients**")
st.dataframe(weights.style.format({"Meta-model coefficient": "{:.4f}"}), use_container_width=True, hide_index=True)
st.caption("Coefficients are learned from training-only out-of-fold predictions. They need not be positive or sum to one; Ridge regression also learns an intercept.")

st.subheader("5. Try a concrete mix")
st.write("Enter values in the same units/ranges as the training data to compare predictions. This is a data-driven estimate, not a substitute for laboratory testing.")
default_values = train[features].median(numeric_only=True)
with st.form("prediction_form"):
    cols = st.columns(2)
    input_values = {}
    for i, feature in enumerate(features):
        with cols[i % 2]:
            input_values[feature] = st.number_input(
                feature,
                value=float(default_values[feature]),
                format="%.3f",
                key=f"input_{i}"
            )
    submitted = st.form_submit_button("Predict compressive strength")
if submitted:
    one_row = pd.DataFrame([[input_values[f] for f in features]], columns=features)
    one_predictions = {name: float(model.predict(one_row)[0]) for name, model in fitted_models.items()}
    combined_value = float(meta_model.predict(np.array([[one_predictions[n] for n in fitted_models.keys()]]))[0])
    result_cols = st.columns(4)
    for col, (name, value) in zip(result_cols, list(one_predictions.items()) + [("Combined", combined_value)]):
        col.metric(name, f"{value:.2f} MPa")

st.subheader("6. Viva preparation")
with st.expander("What is R²?"):
    st.write("R² (coefficient of determination) measures how much variation in the target is explained by the model relative to predicting the test-set mean. 1 is perfect, 0 is no better than that mean baseline, and a negative value is worse than the baseline.")
with st.expander("Why combine three models?"):
    st.write("They have different inductive biases: polynomial regression models explicit polynomial terms, random forests model nonlinear decision rules, and an ANN learns layered nonlinear representations. Stacking can exploit complementary errors.")
with st.expander("Why Ridge Regression as the meta-model?"):
    st.write("Ridge is a linear model with L2 regularization. It learns the contribution of each base model while discouraging overly large coefficients, which can reduce overfitting compared with an unconstrained linear combiner.")
with st.expander("How do you avoid data leakage?"):
    st.write("The final test set is not used to fit the base models' training parameters or the Ridge meta-model. The meta-model uses out-of-fold predictions made within the training set, then the completed stack is evaluated once on the held-out test set.")
with st.expander("Does higher R² prove the combined model is always better?"):
    st.write("No. It demonstrates better fit on this particular test split. Generalization should be confirmed on other data or repeated/nested cross-validation. Do not claim a guaranteed improvement on every dataset.")
with st.expander("What are MAE and RMSE?"):
    st.write("MAE is the average absolute prediction error in MPa. RMSE is the square root of mean squared error, also in MPa, and penalizes large errors more strongly.")
with st.expander("What are the limitations?"):
    st.write("Performance depends on data quality, train/test split, feature coverage, and hyperparameters. Predictions outside the training distribution may be unreliable. The model estimates compressive strength from historical data; laboratory testing remains necessary for engineering acceptance.")

st.subheader("7. Download results")
csv = actual_pred.to_csv(index=False).encode("utf-8")
st.download_button("Download test-set predictions (CSV)", data=csv, file_name="concrete_strength_model_comparison.csv", mime="text/csv")
st.caption("Models are trained when this app starts. Results may vary if the dataset, split, random seeds, or software versions change.")
