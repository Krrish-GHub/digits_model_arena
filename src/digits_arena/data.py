"""Dataset loading and deterministic holdout creation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

SEED = 42
TEST_SIZE = 0.25


@dataclass(frozen=True)
class Dataset:
    X: np.ndarray
    y: np.ndarray
    feature_shape: tuple[int, int]
    name: str


def load_dataset() -> Dataset:
    """Load scikit-learn's bundled 8x8 handwritten digits (pixel values 0..16)."""
    bundle = load_digits()
    # Scale the original 0..16 intensity range to [0, 1]; no data-dependent fit occurs here.
    X = bundle.data.astype(np.float32) / 16.0
    y = bundle.target.astype(np.int64)
    return Dataset(X=X, y=y, feature_shape=(8, 8), name="scikit-learn digits")


def make_holdout(data: Dataset = None):
    """Return one stratified, fixed-seed train/test split; test rows never fit preprocessing."""
    if data is None:
        data = load_dataset()
    return train_test_split(
        data.X,
        data.y,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=data.y,
    )
