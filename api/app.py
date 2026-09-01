from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI
from api.schemas import HeartData

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Heart Disease Prediction API",
    description="Production-style API for heart disease prediction.",
    version="2.0.0",
)


@app.get("/")
def root():
    return {
        "message": "Heart Disease Prediction API",
        "version": "2.0.0",
        "status": "running",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.post("/predict")
def predict(data: HeartData):
    input_data = pd.DataFrame([data.model_dump()])

    prediction = int(model.predict(input_data)[0])

    probability = float(model.predict_proba(input_data)[0][1])

    return {
        "prediction": prediction,
        "probability": round(probability, 4),
    }