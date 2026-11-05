from pathlib import Path

import numpy as np

from digits_arena.benchmark import evaluate_models
from digits_arena.data import load_dataset, make_holdout
from digits_arena.models import model_zoo


def test_digits_dataset_shape_scale_and_labels():
    data = load_dataset()
    assert data.X.shape == (1797, 64)
    assert data.feature_shape == (8, 8)
    assert set(np.unique(data.y)) == set(range(10))
    assert 0 <= data.X.min() <= data.X.max() <= 1


def test_holdout_is_stratified_and_reproducible():
    data = load_dataset()
    first = make_holdout(data)
    second = make_holdout(data)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    X_train, X_test, y_train, y_test = first
    assert len(y_train) == 1347 and len(y_test) == 450
    assert set(np.unique(y_train)) == set(range(10))
    assert set(np.unique(y_test)) == set(range(10))
    assert X_train.shape[1] == X_test.shape[1] == 64


def test_model_zoo_fits_and_predicts():
    data = load_dataset()
    X_train, X_test, y_train, y_test = make_holdout(data)
    for model in model_zoo().values():
        model.fit(X_train, y_train)
        prediction = model.predict(X_test[:8])
        assert prediction.shape == (8,)
        assert set(prediction).issubset(set(range(10)))


def test_benchmark_writes_complete_results(tmp_path: Path):
    result = evaluate_models(tmp_path, timing_repeats=2)
    assert len(result["results"]) == 4
    assert {row["model"] for row in result["results"]} == {
        "logistic_regression", "linear_svm", "rbf_svm", "tiny_mlp_64"
    }
    assert all(0 <= row["accuracy"] <= 1 for row in result["results"])
    assert all(row["serialized_model_bytes_compressed"] > 0 for row in result["results"])
    assert (tmp_path / "benchmark.json").exists()
    assert len(list(tmp_path.glob("confusion_*.csv"))) == 4
