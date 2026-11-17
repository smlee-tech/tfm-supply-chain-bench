"""Metrics, including calibration (ECE) and bootstrap confidence intervals."""
from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    """Binary ECE with equal-width bins on the predicted probability of class 1."""
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    which = np.clip(np.digitize(p, bins[1:-1], right=True), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = which == b
        if mask.any():
            ece += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(ece)


def classification_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    raw = np.asarray(p, dtype=float)
    p = np.clip(raw, 1e-7, 1 - 1e-7)  # log loss needs probabilities away from 0 and 1
    return {
        # Ranking uses the unclipped scores: clipping ties every saturated prediction,
        # which lowers the AUC of an overconfident model and disagrees with bootstrap_ci.
        "roc_auc": float(roc_auc_score(y, raw)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p),
    }


def regression_metrics(y: np.ndarray, yhat: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, yhat))),
        "mae": float(mean_absolute_error(y, yhat)),
        "r2": float(r2_score(y, yhat)),
    }


PRIMARY = {"classification": ("roc_auc", roc_auc_score),
           "regression": ("rmse", lambda y, yh: float(np.sqrt(mean_squared_error(y, yh))))}


def bootstrap_ci(
    y: np.ndarray,
    pred: np.ndarray,
    fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 200,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI of a metric over test rows."""
    y, pred = np.asarray(y), np.asarray(pred)
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            vals.append(fn(y[idx], pred[idx]))
        except ValueError:  # e.g. a resample with one class only
            continue
    if not vals:
        return float("nan"), float("nan")
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
