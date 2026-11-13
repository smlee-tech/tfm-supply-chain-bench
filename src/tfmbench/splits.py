"""Random and temporal train/test splits.

Temporal split: everything before a cutoff date is training data, everything on or
after it is test data. This mirrors how a model is used in practice (train on the
past, predict the future) and is the main difference from IID benchmarks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def random_split(n: int, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(round(n * test_size))
    return np.sort(idx[n_test:]), np.sort(idx[:n_test])


def temporal_split(t: pd.Series, test_size: float) -> tuple[np.ndarray, np.ndarray]:
    """Split so that every training timestamp is strictly before every test timestamp."""
    t = pd.Series(pd.to_datetime(t)).reset_index(drop=True)
    cutoff = t.quantile(1 - test_size)
    train = np.flatnonzero((t < cutoff).to_numpy())
    test = np.flatnonzero((t >= cutoff).to_numpy())
    if len(train) == 0 or len(test) == 0:
        raise ValueError("Temporal split produced an empty side; check the time column.")
    return train, test


def split(mode: str, t: pd.Series, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if mode == "random":
        return random_split(len(t), test_size, seed)
    if mode == "temporal":
        return temporal_split(t, test_size)
    raise ValueError(f"Unknown split mode: {mode}")


def validation_split(
    train_idx: np.ndarray, t: pd.Series, mode: str, val_size: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Carve a validation set out of the training indices, in the same mode as the test split."""
    if mode == "temporal":
        sub_t = pd.Series(pd.to_datetime(t)).iloc[train_idx].reset_index(drop=True)
        tr, va = temporal_split(sub_t, val_size)
    else:
        tr, va = random_split(len(train_idx), val_size, seed)
    return train_idx[tr], train_idx[va]


def subsample(idx: np.ndarray, max_rows: int | None, seed: int) -> np.ndarray:
    """Random subsample of the training window (for learning curves and TFM size limits)."""
    if max_rows is None or len(idx) <= max_rows:
        return idx
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(idx, size=max_rows, replace=False))
