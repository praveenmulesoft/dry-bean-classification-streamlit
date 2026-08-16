"""
Dry Bean Classifier Explorer - Streamlit front-end.

Loads the five pre-trained pipelines from model/ and scores them against
whatever test CSV the user uploads, so the deployed app never trains anything
at request time.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

MODEL_DIR = Path(__file__).resolve().parent / "model"
TARGET_COLUMN = "Class"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest (Ensemble)": "random_forest.joblib",
}

st.set_page_config(
    page_title="Dry Bean Classifier Explorer", page_icon="🫘", layout="wide"
)


@st.cache_resource
def load_artefacts():
    """Load the fitted pipelines and the label encoder once per session."""
    models = {
        name: joblib.load(MODEL_DIR / filename)
        for name, filename in MODEL_FILES.items()
        if (MODEL_DIR / filename).exists()
    }
    encoder = joblib.load(MODEL_DIR / "label_encoder.joblib")
    return models, encoder


@st.cache_data
def load_reference_metrics():
    """Training-time scores, shown as a baseline alongside live results."""
    path = MODEL_DIR / "metrics_summary.csv"
    if path.exists():
        return pd.read_csv(path, index_col="ML Model Name")
    return None


def evaluate(model, encoder, X, y_true_encoded):
    """Score one model on the uploaded data and hand back metrics + predictions."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)

    # A user-supplied slice may not contain every bean variety; one-vs-rest AUC
    # is undefined in that case, so report it as not-applicable rather than
    # crashing the page.
    present = np.unique(y_true_encoded)
    if len(present) < len(encoder.classes_):
        auc = None
    else:
        auc = roc_auc_score(
            y_true_encoded, y_proba, multi_class="ovr", average="macro"
        )

    metrics = {
        "Accuracy": accuracy_score(y_true_encoded, y_pred),
        "AUC": auc,
        "Precision": precision_score(
            y_true_encoded, y_pred, average="macro", zero_division=0
        ),
        "Recall": recall_score(
            y_true_encoded, y_pred, average="macro", zero_division=0
        ),
        "F1": f1_score(y_true_encoded, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_true_encoded, y_pred),
    }
    return metrics, y_pred


def draw_confusion(y_true_encoded, y_pred, class_names, title):
    matrix = confusion_matrix(y_true_encoded, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
        ax=ax,
    )
    ax.set_xlabel("Predicted variety")
    ax.set_ylabel("Actual variety")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    return fig


models, encoder = load_artefacts()
class_names = list(encoder.classes_)

st.title("🫘 Dry Bean Classifier Explorer")
st.caption(
    "Seven varieties of dry bean, separated by sixteen geometric measurements "
    "taken from grain images. Upload a test CSV to score five trained models against it."
)

if not models:
    st.error("No trained models found in model/. Run `python train_models.py` first.")
    st.stop()

with st.sidebar:
    st.header("Controls")
    uploaded = st.file_uploader(
        "Upload test data (CSV)",
        type="csv",
        help="Needs the 16 feature columns plus a 'Class' column of true labels.",
    )
    selected_name = st.selectbox("Model", list(models.keys()))
    compare_all = st.checkbox("Compare all five models", value=False)
    st.divider()
    st.caption(
        "The repo ships `test_data.csv` (2,723 held-out rows) — upload that to "
        "reproduce the README table."
    )

if uploaded is None:
    st.info("⬅️ Upload a test CSV from the sidebar to begin.")
    reference = load_reference_metrics()
    if reference is not None:
        st.subheader("Reference scores from training")
        st.caption("Measured on the held-out 20% split during `train_models.py`.")
        st.dataframe(reference.style.format("{:.4f}"), width="stretch")
    st.stop()

frame = pd.read_csv(uploaded)

if TARGET_COLUMN not in frame.columns:
    st.error(
        f"The uploaded CSV has no '{TARGET_COLUMN}' column, so predictions "
        "cannot be scored against ground truth."
    )
    st.stop()

X = frame.drop(columns=[TARGET_COLUMN])
raw_labels = frame[TARGET_COLUMN]

unknown = set(raw_labels.unique()) - set(class_names)
if unknown:
    st.error(f"Unrecognised class labels in the upload: {sorted(unknown)}")
    st.stop()

y_true = encoder.transform(raw_labels)

st.success(f"Loaded **{len(frame):,} rows** with **{X.shape[1]} features**.")

with st.expander("Preview uploaded data"):
    st.dataframe(frame.head(15), width="stretch")
    st.write("Class distribution:")
    st.bar_chart(raw_labels.value_counts())

if compare_all:
    st.subheader("All models on your uploaded data")
    rows = {}
    for name, model in models.items():
        metrics, _ = evaluate(model, encoder, X, y_true)
        rows[name] = metrics
    table = pd.DataFrame(rows).T
    table.index.name = "ML Model Name"
    st.dataframe(
        table.style.format("{:.4f}", na_rep="n/a").highlight_max(
            axis=0, color="#d4edda"
        ),
        width="stretch",
    )
    st.caption("Best score in each column is shaded green.")
    st.divider()

st.subheader(f"{selected_name} — detailed results")

model = models[selected_name]
metrics, y_pred = evaluate(model, encoder, X, y_true)

cols = st.columns(6)
for col, (label, value) in zip(cols, metrics.items()):
    col.metric(label, "n/a" if value is None else f"{value:.4f}")

if metrics["AUC"] is None:
    st.caption(
        "AUC needs all seven varieties present in the upload; it is reported as "
        "n/a for this subset."
    )

left, right = st.columns([1, 1])

with left:
    st.markdown("**Confusion matrix**")
    st.pyplot(
        draw_confusion(y_true, y_pred, class_names, f"{selected_name}"),
        clear_figure=True,
    )

with right:
    st.markdown("**Classification report**")
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    st.dataframe(
        pd.DataFrame(report).T.style.format("{:.4f}"), width="stretch", height=380
    )

predictions = frame.copy()
predictions["Predicted"] = encoder.inverse_transform(y_pred)
predictions["Correct"] = predictions["Predicted"] == predictions[TARGET_COLUMN]

with st.expander("Row-level predictions"):
    st.dataframe(
        predictions[[TARGET_COLUMN, "Predicted", "Correct"]].head(200),
        width="stretch",
    )
    st.download_button(
        "Download predictions as CSV",
        predictions.to_csv(index=False).encode(),
        file_name=f"predictions_{selected_name.split(' (')[0].lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )
