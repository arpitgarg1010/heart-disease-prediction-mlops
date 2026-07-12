from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI

from api.schemas import HeartData

app = FastAPI(
    title="Heart Disease Prediction API",
    version="1.0"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

model = joblib.load(PROJECT_ROOT / "models" / "best_model.pkl")
scaler = joblib.load(PROJECT_ROOT / "models" / "scaler.pkl")

continuous = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak"
]


@app.get("/")
def home():
    return {
        "message": "Heart Disease Prediction API is running."
    }


@app.post("/predict")
def predict(data: HeartData):

    df = pd.DataFrame([data.model_dump()])

    df[continuous] = scaler.transform(df[continuous])

    prediction = int(model.predict(df)[0])

    probability = float(model.predict_proba(df)[0].max())

    return {
        "prediction": prediction,
        "confidence": round(probability, 4)
    }