"""Train and evaluate baseline churn classifiers without preprocessing leakage."""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_DIR = ROOT / "models"
REPORTS = ROOT / "reports"
MODEL_DIR.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Read README.md for download instructions."
        )

    df = pd.read_csv(DATA_PATH)
    if "Churn" not in df.columns:
        raise ValueError("Expected target column 'Churn' was not found.")

    # customerID identifies a row; it usually should not be a predictive feature.
    df = df.drop(columns=["customerID"], errors="ignore")
    df = df.dropna(subset=["Churn"]).copy()
    y = df["Churn"].map({"No": 0, "Yes": 1})
    valid = y.notna()
    X, y = df.loc[valid].drop(columns="Churn"), y.loc[valid].astype(int)

    # In the common Telco dataset, TotalCharges can be read as text due to blank strings.
    if "TotalCharges" in X.columns:
        X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")

    categorical = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric = X.select_dtypes(include=["number"]).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric),
        ("categorical", categorical_pipeline, categorical),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "decision_tree": DecisionTreeClassifier(
            max_depth=6, random_state=42, class_weight="balanced"
        ),
    }
    results = {}
    best_name, best_f1, best_pipeline = None, -1, None

    for name, estimator in candidates.items():
        pipe = Pipeline([("preprocess", preprocess), ("model", estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        probs = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None
        metrics = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
            "classification_report": classification_report(
                y_test, pred, zero_division=0, output_dict=True
            ),
        }
        if probs is not None and y_test.nunique() == 2:
            metrics["roc_auc"] = float(roc_auc_score(y_test, probs))
        results[name] = metrics
        print(f"\n--- {name} ---")
        print(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"},
                         indent=2))
        if metrics["f1"] > best_f1:
            best_name, best_f1, best_pipeline = name, metrics["f1"], pipe

    joblib.dump(best_pipeline, MODEL_DIR / "churn_pipeline.joblib")
    with open(REPORTS / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"selected_model": best_name, "models": results}, f, indent=2)

    print(f"\nSelected by test-set F1 for this baseline: {best_name}")
    print("Saved model to models/churn_pipeline.joblib")
    print("Saved metrics to reports/metrics.json")
    print("Note: for a stronger final comparison, use cross-validation on training data.")

if __name__ == "__main__":
    main()
