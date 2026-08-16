"""
Dry Bean multi-class classification - model training and evaluation.

Trains five classifiers on the UCI Dry Bean dataset, evaluates each on a
held-out stratified test split, and persists the fitted pipelines plus a
metrics summary for the Streamlit app to consume.

Run:  python train_models.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEED = 42
TEST_FRACTION = 0.20

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_FILE = PROJECT_ROOT / "data" / "dry_bean_full.csv"
MODEL_DIR = PROJECT_ROOT / "model"
TEST_CSV = PROJECT_ROOT / "test_data.csv"
TARGET_COLUMN = "Class"


def build_classifiers():
    """Return the five classifiers, each wrapped so preprocessing travels with it.

    Distance- and gradient-based learners (logistic regression, kNN) are scaled;
    the tree-based learners and GaussianNB are left on the raw feature scale
    because standardisation gains them nothing.
    """
    scaled = lambda estimator: Pipeline(
        [("scaler", StandardScaler()), ("clf", estimator)]
    )

    return {
        "Logistic Regression": scaled(
            LogisticRegression(max_iter=2000, random_state=SEED)
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=12, min_samples_leaf=5, random_state=SEED
        ),
        "kNN": scaled(KNeighborsClassifier(n_neighbors=11, weights="distance")),
        "Naive Bayes": GaussianNB(),
        # Depth and leaf size are capped deliberately: an unbounded forest of 300
        # trees serialises to ~35 MB, which is awkward to keep in git and slow to
        # load on a free-tier dyno. These limits cut the artefact to under 6 MB
        # while giving up only ~0.004 macro-F1.
        "Random Forest (Ensemble)": RandomForestClassifier(
            n_estimators=120,
            max_depth=12,
            min_samples_leaf=8,
            random_state=SEED,
            n_jobs=-1,
        ),
    }


def score_model(model, X_test, y_test):
    """Compute the six required metrics for one fitted model.

    The dataset has seven classes, so precision/recall/F1 are macro-averaged
    (every bean variety counts equally regardless of how many samples it has)
    and AUC uses the one-vs-rest formulation over predicted probabilities.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro"),
        "Precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }


def main():
    frame = pd.read_csv(DATA_FILE)
    features = frame.drop(columns=[TARGET_COLUMN])
    labels = frame[TARGET_COLUMN]

    print(f"Dataset: {frame.shape[0]} instances x {features.shape[1]} features")
    print(f"Classes: {labels.nunique()} -> {sorted(labels.unique())}\n")

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        features, y, test_size=TEST_FRACTION, stratify=y, random_state=SEED
    )
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows\n")

    MODEL_DIR.mkdir(exist_ok=True)
    results = {}

    for name, model in build_classifiers().items():
        model.fit(X_train, y_train)
        metrics = score_model(model, X_test, y_test)
        results[name] = metrics

        slug = name.split(" (")[0].lower().replace(" ", "_")
        joblib.dump(model, MODEL_DIR / f"{slug}.joblib")

        line = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        print(f"{name:26s} {line}")

    joblib.dump(encoder, MODEL_DIR / "label_encoder.joblib")

    # The test split doubles as the CSV students upload in the deployed app,
    # so it carries the original string labels rather than encoded integers.
    test_export = X_test.copy()
    test_export[TARGET_COLUMN] = encoder.inverse_transform(y_test)
    test_export.to_csv(TEST_CSV, index=False)

    summary = pd.DataFrame(results).T.round(4)
    summary.index.name = "ML Model Name"
    summary.to_csv(MODEL_DIR / "metrics_summary.csv")

    with open(MODEL_DIR / "metrics_summary.json", "w") as handle:
        json.dump(results, handle, indent=2)

    winner = max(results, key=lambda k: results[k]["F1"])
    print(f"\nBest macro-F1: {winner} ({results[winner]['F1']:.4f})")
    print(f"Wrote {TEST_CSV.name} ({len(test_export)} rows) and model/ artefacts.")


if __name__ == "__main__":
    main()
