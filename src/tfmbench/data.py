"""Load public datasets and turn them into leakage-safe prediction tasks.

Every task is defined in configs/datasets.yaml. Columns that are only known after
the prediction time are listed as leakage_cols and dropped by default.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


@dataclass
class Task:
    X: pd.DataFrame
    y: pd.Series
    t: pd.Series  # prediction time, used for temporal splits
    task_type: str  # "classification" or "regression"
    name: str
    groups: pd.Series | None = None  # rows that belong together (e.g. one order); grouped split


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_raw(ds_cfg: dict, data_dir: str | Path) -> pd.DataFrame:
    frames = []
    for fname in ds_cfg["files"]:
        fpath = Path(data_dir) / fname
        if not fpath.exists():
            raise FileNotFoundError(
                f"{fpath} not found. Download the dataset first (see README, 'Data')."
            )
        frames.append(pd.read_csv(fpath, low_memory=False, **(ds_cfg.get("read_kwargs") or {})))
    return pd.concat(frames, ignore_index=True)


def _to_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def _target(df: pd.DataFrame, spec: dict) -> tuple[pd.Series, list[str]]:
    """Return the target and the source columns it was built from."""
    kind = spec["kind"]
    if kind == "column":
        return df[spec["col"]], [spec["col"]]
    if kind == "numeric_column":
        return pd.to_numeric(df[spec["col"]], errors="coerce"), [spec["col"]]
    if kind == "late_flag":
        actual = _to_datetime(df[spec["actual"]])
        planned = _to_datetime(df[spec["planned"]])
        y = (actual > planned).astype(float)
        y[actual.isna() | planned.isna()] = np.nan
        return y, [spec["actual"]]
    raise ValueError(f"Unknown target kind: {kind}")


def build_task(
    df: pd.DataFrame,
    ds_cfg: dict,
    task_name: str,
    keep_ids: bool = False,
    inject_leak: bool = False,
) -> Task:
    """Build features, target and prediction time for one task.

    inject_leak=True deliberately keeps the leakage columns. It exists only for the
    leakage-injection ablation, to show how much a leaked column inflates results.
    """
    tcfg = ds_cfg["tasks"][task_name]
    df = df.copy()

    y, target_sources = _target(df, tcfg["target"])
    t = _to_datetime(df[ds_cfg["time_col"]])
    keep = y.notna() & t.notna()
    df, y, t = df[keep], y[keep], t[keep]
    group_col = ds_cfg.get("group_col")
    groups = df[group_col].reset_index(drop=True) if group_col else None

    leakage = [c for c in tcfg.get("leakage_cols", []) if c in df.columns]
    drop = set(target_sources) | set(ds_cfg.get("drop_cols", []))
    if not inject_leak:
        drop |= set(leakage)
    if not keep_ids:
        drop |= set(ds_cfg.get("id_cols", []))

    # Other date columns: drop them, unless they are leakage columns being injected,
    # in which case they become "days since prediction time".
    for col in ds_cfg.get("other_date_cols", []):
        if col not in df.columns:
            continue
        if inject_leak and col in leakage:
            df[col] = (_to_datetime(df[col]) - t).dt.days
        else:
            drop.add(col)

    drop.add(ds_cfg["time_col"])
    X = df.drop(columns=[c for c in drop if c in df.columns])

    # Calendar features from the prediction time (known when predicting).
    X = X.assign(
        pred_month=t.dt.month.values,
        pred_dayofweek=t.dt.day_of_week.values,
        pred_dayofyear=t.dt.day_of_year.values,
    )

    task_type = tcfg["type"]
    if task_type == "classification":
        y = y.astype(int)
    else:
        y = y.astype(float)

    return Task(
        X=X.reset_index(drop=True),
        y=y.reset_index(drop=True),
        t=t.reset_index(drop=True),
        task_type=task_type,
        name=task_name,
        groups=groups,
    )
