from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from src.models.train import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


def evaluate_model(model, X_test, y_test, X, y):
    """Calculate evaluation metrics and 5-fold CV accuracy."""

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "cv_accuracy": cross_val_score(
            model,
            X,
            y,
            cv=5,
            scoring="accuracy",
        ).mean(),
    }

    return metrics


def main():
    X, y = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model_paths = {
        "Logistic Regression": MODELS_DIR / "logistic_regression_pipeline.joblib",
        "Random Forest": MODELS_DIR / "random_forest_pipeline.joblib",
    }

    results = []

    for model_name, model_path in model_paths.items():
        model = joblib.load(model_path)

        metrics = evaluate_model(
            model,
            X_test,
            y_test,
            X,
            y,
        )

        results.append(
            {
                "Model": model_name,
                **metrics,
            }
        )

    results_df = pd.DataFrame(results)

    print("\nModel Evaluation Results:")
    print(results_df.to_string(index=False))

    results_df.to_csv(
        REPORTS_DIR / "metrics.csv",
        index=False,
    )

    print("\nMetrics saved to:", REPORTS_DIR / "metrics.csv")


if __name__ == "__main__":
    main()