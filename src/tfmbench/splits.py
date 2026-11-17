"""Random, grouped and temporal train/test splits.

- random: rows are shuffled. Rows of the same group (e.g. lines of one order) can land
  on both sides, so the score also rewards memorising groups.
- grouped: whole groups are shuffled; no group is on both sides. Still IID in time.
  random - grouped = effect of group leakage; grouped - temporal = effect of time.
- temporal: everything before a cutoff is training data, everything on or after it is
  test data. Evaluated at several cutoffs (rolling origin, expanding training window),
  because one cutoff is a single draw of "the future".
"""
from __future__ import annotations

from dataclasses import dataclass

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


def grouped_split(groups: pd.Series, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Random split of whole groups: no group has rows on both sides.

    About test_size of the groups (not exactly of the rows) go to the test side.
    Rows with a missing group are treated as their own group.
    """
    codes = pd.factorize(pd.Series(groups).reset_index(drop=True))[0].astype(np.int64)
    missing = codes < 0
    codes[missing] = codes.max() + 1 + np.arange(missing.sum())
    uniq = np.unique(codes)
    rng = np.random.default_rng(seed)
    n_test = int(round(len(uniq) * test_size))
    test_groups = rng.permutation(uniq)[:n_test]
    is_test = np.isin(codes, test_groups)
    train, test = np.flatnonzero(~is_test), np.flatnonzero(is_test)
    if len(train) == 0 or len(test) == 0:
        raise ValueError("Grouped split produced an empty side; check the group column.")
    return train, test


@dataclass
class Fold:
    fold: int
    train: np.ndarray
    test: np.ndarray
    cutoff: pd.Timestamp | None = None  # temporal folds only


def temporal_folds(t: pd.Series, test_size: float, n_folds: int) -> list[Fold]:
    """Rolling-origin folds with an expanding training window.

    The last n_folds * test_size of the timeline is cut into n_folds consecutive test
    blocks. Fold k trains on everything strictly before its block. The last fold is
    identical to temporal_split(t, test_size).
    """
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    if n_folds * test_size >= 1:
        raise ValueError(f"n_folds * test_size must be < 1 (got {n_folds} * {test_size}).")
    t = pd.Series(pd.to_datetime(t)).reset_index(drop=True)
    folds = []
    for k in range(n_folds):
        start_q = 1 - (n_folds - k) * test_size
        cutoff = t.quantile(start_q)
        in_test = t >= cutoff
        if k < n_folds - 1:
            in_test &= t < t.quantile(start_q + test_size)
        train = np.flatnonzero((t < cutoff).to_numpy())
        test = np.flatnonzero(in_test.to_numpy())
        if len(train) == 0 or len(test) == 0:
            raise ValueError(f"Temporal fold {k} has an empty side; check the time column.")
        folds.append(Fold(k, train, test, cutoff))
    return folds


def make_folds(
    mode: str,
    t: pd.Series,
    groups: pd.Series | None,
    test_size: float,
    seed: int,
    n_folds: int = 1,
) -> list[Fold]:
    """All train/test folds for one seed. Only the temporal mode has more than one."""
    if mode == "random":
        return [Fold(0, *random_split(len(t), test_size, seed))]
    if mode == "grouped":
        return [Fold(0, *grouped_split(_require_groups(groups), test_size, seed))]
    if mode == "temporal":
        return temporal_folds(t, test_size, n_folds)
    raise ValueError(f"Unknown split mode: {mode}")


def _require_groups(groups: pd.Series | None) -> pd.Series:
    if groups is None:
        raise ValueError("The grouped split needs a group column: set group_col for the "
                         "dataset in configs/datasets.yaml.")
    return groups


def validation_split(
    train_idx: np.ndarray,
    t: pd.Series,
    mode: str,
    val_size: float,
    seed: int,
    groups: pd.Series | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Carve a validation set out of the training indices, in the same mode as the test split."""
    if mode == "temporal":
        sub_t = pd.Series(pd.to_datetime(t)).iloc[train_idx].reset_index(drop=True)
        tr, va = temporal_split(sub_t, val_size)
    elif mode == "grouped":
        tr, va = grouped_split(_require_groups(groups).iloc[train_idx], val_size, seed)
    else:
        tr, va = random_split(len(train_idx), val_size, seed)
    return train_idx[tr], train_idx[va]


LABEL_POLICIES = ("oracle", "known", "gap")


def label_available(
    train_idx: np.ndarray,
    t: pd.Series,
    label_t: pd.Series | None,
    cutoff: pd.Timestamp,
    policy: str,
    gap_days: float = 30.0,
) -> np.ndarray:
    """Training rows a model could really use at the cutoff, given when labels arrive.

    A label is not known at prediction time but at label_t (an order is only known to be
    on time once it is delivered). When that delay depends on the outcome, the rows
    with a known label shortly before the cutoff are a biased sample.

    - oracle: every row before the cutoff, as if labels were instant (the usual temporal
      split; optimistic).
    - known: rows whose label is known before the cutoff. Recent but biased.
    - gap: as "known", and also without the last gap_days before the cutoff, so that
      nearly every remaining row has its label. Less biased but older.
    """
    if policy not in LABEL_POLICIES:
        raise ValueError(f"Unknown label policy: {policy}")
    if policy == "oracle":
        return train_idx
    if label_t is None:
        raise ValueError("This label policy needs label_time_cols for the task in "
                         "configs/datasets.yaml.")
    cutoff = pd.Timestamp(cutoff)
    lt = pd.Series(pd.to_datetime(label_t)).iloc[train_idx]
    keep = (lt < cutoff).to_numpy()  # NaT compares False: never known
    if policy == "gap":
        tt = pd.Series(pd.to_datetime(t)).iloc[train_idx]
        keep = keep & (tt < cutoff - pd.Timedelta(days=gap_days)).to_numpy()
    if not keep.any():
        raise ValueError(f"Label policy {policy!r} leaves no training rows before {cutoff}.")
    return train_idx[keep]


TRAIN_SAMPLES = ("random", "recent")


def subsample(idx: np.ndarray, max_rows: int | None, seed: int, how: str = "random",
              t: pd.Series | None = None) -> np.ndarray:
    """Cut the training window down to max_rows (learning curves, TFM size limits).

    random: a uniform sample of the window. recent: the max_rows rows with the latest
    prediction time, i.e. a context refreshed with the newest usable data (needs t).
    """
    if how not in TRAIN_SAMPLES:
        raise ValueError(f"Unknown train sample: {how}")
    if max_rows is None or len(idx) <= max_rows:
        return idx
    if how == "recent":
        if t is None:
            raise ValueError("train sample 'recent' needs the prediction times.")
        order = np.argsort(pd.Series(pd.to_datetime(t)).iloc[idx].to_numpy(), kind="stable")
        return np.sort(idx[order[-max_rows:]])
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(idx, size=max_rows, replace=False))
