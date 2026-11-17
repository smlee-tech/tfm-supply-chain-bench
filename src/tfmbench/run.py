"""Run one experiment grid and append results to a CSV.

Example:
  tfmbench --dataset dataco --task late_delivery --split temporal \
           --models linear xgboost lightgbm catboost tabpfn tabicl --seeds 0 1 2 3 4

Cost columns: tune_s is the hyperparameter search (0 for untuned models), fit_s the final
fit, predict_s the test prediction. Use --n-trials 0 for GBDTs at default settings.
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
from .splits import (LABEL_POLICIES, TRAIN_SAMPLES, label_available, make_folds, subsample,
                     validation_split)


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
    n_folds: int = 1,
    meta: dict | None = None,
    log=print,
    label_policy: str = "oracle",
    label_gap_days: float = 30.0,
    train_sample: str = "random",
) -> pd.DataFrame:
    if label_policy != "oracle" and split_mode != "temporal":
        raise ValueError("label_policy applies to the temporal split only.")
    rows = []
    X, y, t = task.X, task.y.to_numpy(), task.t
    for seed in seeds:
        for fold in make_folds(split_mode, t, task.groups, test_size, seed, n_folds):
            usable = label_available(fold.train, t, task.label_t, fold.cutoff, label_policy,
                                     label_gap_days)
            train_idx = subsample(usable, max_train_rows, seed, train_sample, t)
            test_idx = fold.test
            fit_idx, val_idx = validation_split(train_idx, t, split_mode, val_size, seed,
                                                task.groups)
            for name in models:
                spec = REGISTRY[name]
                pre = make_preprocessor(spec.family, X.iloc[train_idx])
                cat_idx = categorical_indices(X) if spec.family == "tree" else []
                try:
                    params, tune_s = {}, 0.0
                    if spec.space is not None and n_trials > 0:
                        pre_tune = make_preprocessor(spec.family, X.iloc[fit_idx])
                        Xf = to_float(pre_tune.fit_transform(X.iloc[fit_idx]))
                        Xv = to_float(pre_tune.transform(X.iloc[val_idx]))
                        t0 = time.perf_counter()
                        params = tune(spec, task.task_type, seed, cat_idx, Xf, y[fit_idx], Xv,
                                      y[val_idx], n_trials)
                        tune_s = time.perf_counter() - t0
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
                    "model": name, "seed": seed, "fold": fold.fold,
                    "cutoff": fold.cutoff, "max_train_rows": max_train_rows,
                    "train_sample": train_sample,
                    "label_policy": label_policy,
                    "label_gap_days": label_gap_days if label_policy == "gap" else None,
                    "train_target_mean": round(float(y[train_idx].mean()), 5),
                    "n_train": len(train_idx), "n_test": len(test_idx), "n_features": X.shape[1],
                    "tune_s": round(tune_s, 3), "fit_s": round(fit_s, 3),
                    "predict_s": round(pred_s, 3),
                    **m, f"{pname}_ci_lo": lo, f"{pname}_ci_hi": hi,
                    "tuned_params": params or None,
                })
                log(f"seed={seed} fold={fold.fold} {name:9s} {pname}={m[pname]:.4f} "
                    f"[{lo:.4f}, {hi:.4f}] tune={tune_s:.1f}s fit={fit_s:.1f}s")
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    metric = PRIMARY[df["task_type"].iloc[0]][0]
    # Seeds first, folds second: under the temporal split every seed sees the same test
    # rows, so the spread over seeds says nothing about how stable a result is over time.
    per_fold = df.groupby(["split", "model", "fold"]).agg(
        m=(metric, "mean"), s=(metric, "std"), tune_s=("tune_s", "mean"), fit_s=("fit_s", "mean"))
    out = per_fold.groupby(["split", "model"]).agg(**{
        metric: ("m", "mean"),
        "std_over_folds": ("m", "std"),
        "std_over_seeds": ("s", "mean"),
        "tune_s": ("tune_s", "mean"),
        "fit_s": ("fit_s", "mean"),
    })
    return out.round(4)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default="configs/datasets.yaml")
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--split", choices=["random", "grouped", "temporal"], default="temporal",
                    help="random: rows shuffled; grouped: whole groups (e.g. orders) shuffled;\n"
                         "temporal: train on the past, test on the future")
    ap.add_argument("--n-folds", type=int, default=3,
                    help="temporal split only: number of rolling-origin cutoffs")
    ap.add_argument("--models", nargs="+", default=["linear", "xgboost", "lightgbm", "catboost"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--max-train-rows", type=int, default=None)
    ap.add_argument("--n-trials", type=int, default=50, help="Optuna trials per GBDT")
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--keep-ids", action="store_true", help="ablation: keep high-cardinality IDs")
    ap.add_argument("--inject-leak", action="store_true",
                    help="ablation: deliberately keep leakage columns")
    ap.add_argument("--drop-cols", nargs="*", default=[],
                    help="ablation: feature columns to leave out, e.g. pred_month pred_dayofyear")
    ap.add_argument("--train-sample", choices=TRAIN_SAMPLES, default="random",
                    help="how --max-train-rows cuts the training window. random: uniform\n"
                         "sample (default); recent: the latest rows, as a refreshed context")
    ap.add_argument("--label-policy", choices=LABEL_POLICIES, default="oracle",
                    help="temporal split only: which training labels count as known at the\n"
                         "cutoff. oracle: all (default); known: only labels that have arrived;\n"
                         "gap: known, and without the last --label-gap-days before the cutoff")
    ap.add_argument("--label-gap-days", type=float, default=30.0)
    ap.add_argument("--out", default=None,
                    help="default: results/<dataset>_<task>.csv, one file per task, because\n"
                         "classification and regression rows have different columns")
    args = ap.parse_args(argv)

    if args.label_policy != "oracle" and args.split != "temporal":
        ap.error("--label-policy applies to --split temporal only")
    unknown = [m for m in args.models if m not in REGISTRY]
    if unknown:
        ap.error(f"unknown models: {unknown}; choose from {list(REGISTRY)}")

    cfg = data_mod.load_config(args.config)
    ds_cfg = cfg["datasets"][args.dataset]
    raw = data_mod.load_raw(ds_cfg, args.data_dir)
    task = data_mod.build_task(raw, ds_cfg, args.task, keep_ids=args.keep_ids,
                               inject_leak=args.inject_leak)
    missing = [c for c in args.drop_cols if c not in task.X.columns]
    if missing:
        ap.error(f"--drop-cols: not a feature column: {missing}; features are {list(task.X.columns)}")
    task.X = task.X.drop(columns=args.drop_cols)
    print(f"{args.dataset}/{args.task}: {len(task.y)} rows, {task.X.shape[1]} features")

    meta = {"dataset": args.dataset, "keep_ids": args.keep_ids, "inject_leak": args.inject_leak,
            "drop_cols": " ".join(sorted(args.drop_cols))}
    df = run_experiment(task, args.split, args.models, args.seeds, args.test_size,
                        max_train_rows=args.max_train_rows, n_trials=args.n_trials,
                        n_boot=args.n_boot, n_folds=args.n_folds, meta=meta,
                        label_policy=args.label_policy, label_gap_days=args.label_gap_days,
                        train_sample=args.train_sample)
    out = Path(args.out or f"results/{args.dataset}_{args.task}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and list(pd.read_csv(out, nrows=0).columns) != list(df.columns):
        ap.error(f"{out} has different columns (another task type or an older run); "
                 "choose another --out")
    df.to_csv(out, mode="a", header=not out.exists(), index=False)
    print(summarize(df).to_string())
    print(f"appended {len(df)} rows to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
