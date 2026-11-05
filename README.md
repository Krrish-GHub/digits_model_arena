# Digits Model Arena

A small, reproducible vision benchmark: compare four compact classifiers on the handwritten-digits dataset shipped with scikit-learn. The point is not to crown a universal winner, but to make the accuracy, model size, inference cost and error patterns visible under one controlled split.

## Why this project

Tiny image classification is a clean way to study core ML trade-offs before moving to heavier image datasets or neural-network frameworks. This benchmark compares a linear probabilistic baseline, a linear SVM, an RBF SVM and a one-hidden-layer MLP. It keeps feature scaling inside each training pipeline, uses a stratified fixed-seed holdout, and reports per-class behavior as well as aggregate scores.

## Dataset and limits

The benchmark uses `sklearn.datasets.load_digits`: 1,797 labeled 8×8 grayscale digit images, with pixel values 0–16 and ten classes. The data are bundled with scikit-learn, so the benchmark does not need a network download. Pixel intensities are divided by 16 using a fixed transform. See the [scikit-learn dataset documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html).

This is a small teaching benchmark—not a claim about phone-camera, handwriting-in-the-wild, or production performance. Its tiny images and sample count make it useful for controlled comparisons, not broad vision conclusions.

## Models and protocol

- Logistic regression: scaled input, interpretable linear baseline with probabilities.
- Linear SVM: scaled linear margin classifier.
- RBF SVM: scaled nonlinear kernel model.
- Tiny MLP: one 64-unit hidden layer with early stopping using only a validation slice of the training split.

One seeded, stratified 75/25 train/test split is used. Scalers are fitted inside sklearn pipelines on training data only. The holdout is used for final metrics and is not used to select model parameters. Reported metrics include accuracy, macro-F1, weighted-F1, per-class precision/recall/F1, compressed serialized estimator size, fit time and median repeated whole-holdout inference time per image. Timing and compressed file size are descriptive measurements from the machine that ran the benchmark, not universal deployment guarantees.

## Run it

```bash
python -m venv .venv
. .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -e '.[test]'
python -m pytest -q
python -m digits_arena.benchmark --output results
```

Or install with `uv sync --extra test` and run `uv run digits-arena --output results`.

The generated `results/benchmark.json` records the dataset, split, library versions, metrics and caveats. `results/confusion_<model>.csv` contains each model's 10×10 confusion matrix. The committed `results/` snapshot is the run used for this README; rerun locally to see measurements on your machine.

## Snapshot from the committed run

The fixed 450-image holdout gave both SVMs 98.0% accuracy; logistic regression and the tiny MLP each reached 97.8%. The compact linear model was the smallest and fastest here, while the RBF SVM used a larger serialized artifact and had higher per-image latency. This is one split and one machine, not a general model ranking.

| Model | Accuracy | Macro-F1 | Compressed model | Median inference / image |
|---|---:|---:|---:|---:|
| Logistic regression | 97.8% | 0.9776 | 4.7 KB | 0.69 µs |
| Linear SVM | 98.0% | 0.9798 | 55.6 KB | 9.06 µs |
| RBF SVM | 98.0% | 0.9799 | 118.7 KB | 39.29 µs |
| Tiny MLP (64 hidden units) | 97.8% | 0.9777 | 61.0 KB | 1.19 µs |

Measurements are from the recorded Python 3.14.7 / scikit-learn 1.9.1 run; timing is especially machine-dependent. Full metrics and per-class errors are in the committed JSON and confusion-matrix CSVs.

## Reading the results

Compare macro-F1 with accuracy to see whether a strong aggregate score hides weaker classes. Inspect the confusion matrices and per-class metrics to find which digits are mixed up. Compare fit time and model artifact size with test accuracy before making a deployment choice. Inference timings are especially sensitive to hardware, background load and library versions, so avoid interpreting tiny differences as a stable ranking.

## Reproducibility and project scope

- Fixed seed: 42.
- No external image files or personal data.
- All estimators and preprocessing are recreated from source; results are plain JSON/CSV.
- GitHub Actions runs the tests on pushes and pull requests.

The benchmark does not claim generalization beyond this dataset, and this implementation does not demonstrate GPU inference or production serving.

## License

Project code is MIT-licensed. The digits examples are distributed with scikit-learn; consult the linked scikit-learn documentation and project license for dataset/source terms.

## Live demo

Try the [interactive GitHub Pages demo](https://krrish-ghub.github.io/digits-model-arena/).
