from fastapi.testclient import TestClient

from api.app import app

client = TestClient(app)


def test_predict_valid_request():
    payload = {
        "age": 63,
        "sex": 1,
        "cp": 1,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 2,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 3,
        "ca": 0,
        "thal": 6,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    body = response.json()

    assert "prediction" in body
    assert "confidence" in body
    assert body["prediction"] in [0, 1]
    assert 0 <= body["confidence"] <= 1


def test_predict_invalid_request():
    response = client.post("/predict", json={})

    assert response.status_code == 422