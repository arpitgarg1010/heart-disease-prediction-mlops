from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI
from api.schemas import HeartData
from prometheus_fastapi_instrumentator import Instrumentator

import logging
import time

from prometheus_client import Counter

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Heart Disease Prediction API",
    description="Production-style API for heart disease prediction.",
    version="2.0.0",
)

Instrumentator().instrument(app).expose(app)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("heart-disease-api")

predictions_total = Counter(
    "predictions_total",
    "Total number of heart disease predictions",
)

predictions_by_class_total = Counter(
    "predictions_by_class_total",
    "Total predictions by predicted class",
    ["prediction"],
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
    start_time = time.perf_counter()

    input_data = pd.DataFrame([data.model_dump()])

    prediction = int(model.predict(input_data)[0])
    confidence = float(model.predict_proba(input_data)[0][1])

    predictions_total.inc()
    predictions_by_class_total.labels(prediction=str(prediction)).inc()

    elapsed = time.perf_counter() - start_time

    logger.info(
        "prediction_completed | prediction=%s | latency_ms=%.2f",
        prediction,
        elapsed * 1000,
    )

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
    }