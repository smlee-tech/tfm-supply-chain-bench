"""Run one experiment grid and append results to a CSV.

Example:
  tfmbench --dataset dataco --task late_delivery --split temporal \
           --models linear xgboost lightgbm catboost tabpfn tabicl --seeds 0 1 2 3 4
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as data_mod
from .metrics import PRIMARY, bootstrap_ci, classification_metrics, regression_metrics
from .models import REGISTRY, ModelUnavailable, predict_scores, tune
from .preprocess import categorical_indices, make_preprocessor, to_float
from .splits import split, subsample, validation_split


def run_experiment(
    task: data_mod.Task,
    split_mode: str,
    models: list[str],
    seeds: list[int],
    test_size: float = 0.2,
    val_size: float = 0.2,
    max_train_rows: int | None = None,
    n_trials: int = 50,
    n_boot: int = 200,
    meta: dict | None = None,
    log=print,
) -> pd.DataFrame:
    rows = []
    X, y, t = task.X, task.y.to_numpy(), task.t
    for seed in seeds:
        train_idx, test_idx = split(split_mode, t, test_size, seed)
        train_idx = subsample(train_idx, max_train_rows, seed)
        fit_idx, val_idx = validation_split(train_idx, t, split_mode, val_size, seed)
        for name in models:
            spec = REGISTRY[name]
            pre = make_preprocessor(spec.family, X.iloc[train_idx])
            cat_idx = categorical_indices(X) if spec.family == "tree" else []
            try:
                params = {}
                if spec.space is not None and n_trials > 0:
                    pre_tune = make_preprocessor(spec.family, X.iloc[fit_idx])
                    Xf = to_float(pre_tune.fit_transform(X.iloc[fit_idx]))
                    Xv = to_float(pre_tune.transform(X.iloc[val_idx]))
                    params = tune(spec, task.task_type, seed, cat_idx, Xf, y[fit_idx], Xv,
                                  y[val_idx], n_trials)
                Xtr = to_float(pre.fit_transform(X.iloc[train_idx]))
                Xte = to_float(pre.transform(X.iloc[test_idx]))
                model = spec.build(task.task_type, seed, params, cat_idx)
                t0 = time.perf_counter()
                model.fit(Xtr, y[train_idx])
                fit_s = time.perf_counter() - t0
                t0 = time.perf_counter()
                scores = predict_scores(model, Xte, task.task_type)
                pred_s = time.perf_counter() - t0
            except ModelUnavailable as e:
                log(f"[skip] {name}: {e}")
                continue

            y_te = y[test_idx]
            if task.task_type == "classification":
                m = classification_metrics(y_te, scores)
            else:
                m = regression_metrics(y_te, scores)
            pname, pfn = PRIMARY[task.task_type]
            lo, hi = bootstrap_ci(y_te, scores, pfn, n_boot=n_boot, seed=seed)
            rows.append({
                **(meta or {}),
                "task": task.name, "task_type": task.task_type, "split": split_mode,
                "model": name, "seed": seed,
                "n_train": len(train_idx), "n_test": len(test_idx), "n_features": X.shape[1],
                "fit_s": round(fit_s, 3), "predict_s": round(pred_s, 3),
                **m, f"{pname}_ci_lo": lo, f"{pname}_ci_hi": hi,
                "tuned_params": params or None,
            })
            log(f"seed={seed} {name:9s} {pname}={m[pname]:.4f} [{lo:.4f}, {hi:.4f}] "
                f"fit={fit_s:.1f}s")
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    metric = PRIMARY[df["task_type"].iloc[0]][0]
    g = df.groupby(["split", "model", "n_train"])[[metric, "fit_s"]]
    out = g.agg(["mean", "std"]).round(4)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default="configs/datasets.yaml")
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--split", choices=["random", "temporal"], default="temporal")
    ap.add_argument("--models", nargs="+", default=["linear", "xgboost", "lightgbm", "catboost"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--max-train-rows", type=int, default=None)
    ap.add_argument("--n-trials", type=int, default=50, help="Optuna trials per GBDT")
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--keep-ids", action="store_true", help="ablation: keep high-cardinality IDs")
    ap.add_argument("--inject-leak", action="store_true",
                    help="ablation: deliberately keep leakage columns")
    ap.add_argument("--out", default="results/results.csv")
    args = ap.parse_args(argv)

    unknown = [m for m in args.models if m not in REGISTRY]
    if unknown:
        ap.error(f"unknown models: {unknown}; choose from {list(REGISTRY)}")

    cfg = data_mod.load_config(args.config)
    ds_cfg = cfg["datasets"][args.dataset]
    raw = data_mod.load_raw(ds_cfg, args.data_dir)
    task = data_mod.build_task(raw, ds_cfg, args.task, keep_ids=args.keep_ids,
                               inject_leak=args.inject_leak)
    print(f"{args.dataset}/{args.task}: {len(task.y)} rows, {task.X.shape[1]} features")

    meta = {"dataset": args.dataset, "keep_ids": args.keep_ids, "inject_leak": args.inject_leak}
    df = run_experiment(task, args.split, args.models, args.seeds, args.test_size,
                        max_train_rows=args.max_train_rows, n_trials=args.n_trials,
                        n_boot=args.n_boot, meta=meta)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, mode="a", header=not out.exists(), index=False)
    print(summarize(df).to_string())
    print(f"appended {len(df)} rows to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
