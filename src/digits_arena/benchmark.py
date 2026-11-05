"""Run a reproducible compact-classifier comparison and write machine-readable artifacts."""
from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .data import SEED, load_dataset, make_holdout
from .models import model_zoo


def evaluate_models(output_dir: Path, timing_repeats: int = 7) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_dataset()
    X_train, X_test, y_train, y_test = make_holdout(data)
    results = []
    per_class: dict[str, dict] = {}

    for name, model in model_zoo().items():
        fit_start = time.perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - fit_start
        predictions = model.predict(X_test)
        probs_available = hasattr(model, "predict_proba")

        # Warm caches, then time repeated whole-holdout predictions; report median per image.
        model.predict(X_test)
        timings = []
        for _ in range(timing_repeats):
            start = time.perf_counter()
            model.predict(X_test)
            timings.append(time.perf_counter() - start)
        median_batch_seconds = statistics.median(timings)
        per_image_us = median_batch_seconds / len(X_test) * 1_000_000

        with tempfile.NamedTemporaryFile(suffix=".joblib") as artifact:
            joblib.dump(model, artifact.name, compress=3)
            artifact.seek(0, 2)
            model_bytes = artifact.tell()

        report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
        per_class[name] = {str(label): report[str(label)] for label in range(10)}
        matrix = confusion_matrix(y_test, predictions, labels=list(range(10)))
        results.append(
            {
                "model": name,
                "accuracy": float(accuracy_score(y_test, predictions)),
                "macro_f1": float(f1_score(y_test, predictions, average="macro")),
                "weighted_f1": float(f1_score(y_test, predictions, average="weighted")),
                "fit_seconds": fit_seconds,
                "median_inference_us_per_image": per_image_us,
                "serialized_model_bytes_compressed": model_bytes,
                "max_iterations": (
                    int(np.max(getattr(model[-1], "n_iter_")))
                    if isinstance(getattr(model[-1], "n_iter_", None), np.ndarray)
                    else (
                        int(getattr(model[-1], "n_iter_"))
                        if getattr(model[-1], "n_iter_", None) is not None
                        else None
                    )
                ),
                "probability_output": probs_available,
            }
        )
        with (output_dir / f"confusion_{name}.csv").open("w", newline="") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(["true/predicted", *range(10)])
            for label, row in enumerate(matrix):
                writer.writerow([label, *row.tolist()])

    results.sort(key=lambda row: (-row["accuracy"], row["median_inference_us_per_image"]))
    payload = {
        "dataset": {
            "name": data.name,
            "source": "sklearn.datasets.load_digits (bundled with scikit-learn)",
            "samples": len(data.y),
            "features": int(data.X.shape[1]),
            "image_shape": list(data.feature_shape),
            "classes": sorted(int(x) for x in np.unique(data.y)),
            "pixel_scale": "original integer range 0..16 divided by 16; fixed transform",
        },
        "protocol": {
            "seed": SEED,
            "test_fraction": 0.25,
            "train_samples": len(y_train),
            "test_samples": len(y_test),
            "split": "stratified random holdout; scalers are fit only inside each training pipeline",
            "latency_repeats": timing_repeats,
            "latency_summary": "median duration for batch prediction divided by holdout size; same machine, one run",
        },
        "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__, "numpy": np.__version__},
        "results": results,
        "per_class_metrics": per_class,
        "notes": [
            "Timing is hardware-, load-, and software-dependent; it is descriptive, not a universal ranking.",
            "Serialized estimator size is compressed joblib size, not peak RAM or deployment package size.",
            "This small 8x8 digits dataset is a teaching benchmark; results do not establish performance on modern camera images.",
        ],
    }
    (output_dir / "benchmark.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--timing-repeats", type=int, default=7)
    args = parser.parse_args()
    result = evaluate_models(args.output, args.timing_repeats)
    print(json.dumps(result["results"], indent=2))
    print(f"\nWrote results to {args.output.resolve()}")


if __name__ == "__main__":
    main()
