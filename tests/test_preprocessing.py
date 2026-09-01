import pandas as pd

from src.features.preprocessing import build_preprocessor


def test_preprocessor_handles_missing_values():
    data = pd.DataFrame(
        {
            "age": [63, 67, None],
            "sex": [1, 1, 0],
            "cp": [1, 4, 2],
            "trestbps": [145, 160, 130],
            "chol": [233, 286, 204],
            "fbs": [1, 0, 0],
            "restecg": [2, 2, 0],
            "thalach": [150, 108, 172],
            "exang": [0, 1, 0],
            "oldpeak": [2.3, 1.5, 1.4],
            "slope": [3, 2, 1],
            "ca": [0, None, 0],
            "thal": [6, 3, None],
        }
    )

    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(data)

    assert transformed.shape[0] == 3
    assert not pd.isna(transformed).any()