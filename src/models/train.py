from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from src.features.preprocessing import build_preprocessor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "heart.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


def load_dataset():
    """Load the heart-disease dataset and create a binary target."""
    df = pd.read_csv(DATA_PATH)

    df["target"] = (df["num"] > 0).astype(int)
    df = df.drop(columns=["num"])

    X = df.drop(columns=["target"])
    y = df["target"]

    return X, y


def build_models():
    """Create Logistic Regression and tuned Random Forest pipelines."""

    preprocessor = build_preprocessor()

    logistic_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=5000,
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )

    random_forest_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "model",
                RandomForestClassifier(
                    random_state=42,
                ),
            ),
        ]
    )

    rf_params = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 5, 10],
    }

    tuned_rf = GridSearchCV(
        random_forest_pipeline,
        param_grid=rf_params,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )

    return logistic_pipeline, tuned_rf


def train_models():
    """Train both models and save the fitted pipelines."""

    X, y = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    logistic_pipeline, tuned_rf = build_models()

    logistic_pipeline.fit(X_train, y_train)
    tuned_rf.fit(X_train, y_train)

    joblib.dump(
        logistic_pipeline,
        MODELS_DIR / "logistic_regression_pipeline.joblib",
    )

    joblib.dump(
        tuned_rf.best_estimator_,
        MODELS_DIR / "random_forest_pipeline.joblib",
    )

    joblib.dump(
        tuned_rf.best_estimator_,
        MODELS_DIR / "best_model.joblib",
    )

    return (
        logistic_pipeline,
        tuned_rf.best_estimator_,
        X_test,
        y_test,
        tuned_rf.best_params_,
    )


if __name__ == "__main__":
    _, _, _, _, best_params = train_models()
    print("Training completed successfully.")
    print("Best Random Forest parameters:", best_params)