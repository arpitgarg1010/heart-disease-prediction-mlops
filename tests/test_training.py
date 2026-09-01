from src.models.train import load_dataset, build_models


def test_dataset_loads():
    X, y = load_dataset()

    assert len(X) > 0
    assert len(y) == len(X)
    assert "target" not in X.columns
    assert set(y.unique()).issubset({0, 1})


def test_models_can_be_created():
    logistic_model, random_forest_model = build_models()

    assert logistic_model is not None
    assert random_forest_model is not None