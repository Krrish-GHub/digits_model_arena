"""Small, interpretable classifier families with leakage-safe preprocessing."""
from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def model_zoo(seed: int = 42):
    """Build fresh deterministic estimators. Scalers fit on training rows inside each pipeline."""
    return {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", random_state=seed),
        ),
        "linear_svm": make_pipeline(
            StandardScaler(),
            SVC(kernel="linear", C=1.0),
        ),
        "rbf_svm": make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=5.0, gamma="scale"),
        ),
        "tiny_mlp_64": make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(64,),
                activation="relu",
                solver="adam",
                alpha=1e-3,
                batch_size=64,
                learning_rate_init=1e-3,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=20,
                random_state=seed,
            ),
        ),
    }
