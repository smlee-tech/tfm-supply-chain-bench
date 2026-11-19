"""Check experiment: how far back should the training window (context) reach when labels
arrive late?

For one dataset, the training rows are the usable rows (by label policy) of the last
W days before the latest usable row, cut to --max-train-rows by a random sample. W is
given in multiples of the label delay G of the task (--gap-days), so W/G is comparable
across datasets. W/G = inf uses the whole history.

Runs linear and LightGBM at default settings over the rolling-origin cutoffs, with all
features and without the calendar features. Results append to
results/label_delay/window_<dataset>_<task>.csv.

Usage: python scripts/context_window.py --dataset olist --task late_delivery --gap-days 30
       python scripts/context_window.py ... --models tabpfn --seeds 0 --ratios 3 --feats nodate
(run TabPFN in its own call: on macOS it must not share a process with LightGBM)

A configuration is skipped when a cutoff leaves a single class in the window (with
`known`, a window shorter than the label delay holds only the outcome that is labelled
first).

The window is applied by wrapping tfmbench.run.label_available for this process only; it
is not a tfmbench option yet.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd

import tfmbench.run as run_mod
from tfmbench import data as data_mod
from tfmbench.splits import label_available

CALENDAR = ["pred_month", "pred_dayofyear", "pred_dayofweek"]


def windowed(window_days: float):
    """label_available, then only the last window_days before the latest usable row."""
    def f(train_idx, t, label_t, cutoff, policy, gap_days=30.0):
        idx = label_available(train_idx, t, label_t, cutoff, policy, gap_days)
        if np.isinf(window_days):
            return idx
        tt = pd.Series(pd.to_datetime(t)).iloc[idx]
        return idx[(tt >= tt.max() - pd.Timedelta(days=window_days)).to_numpy()]
    return f


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--gap-days", type=float, required=True, help="label delay G of the task")
    ap.add_argument("--ratios", nargs="+", type=float, default=[0.25, 0.5, 1, 1.25, 1.5, 2, 3, 4, 8, np.inf])
    ap.add_argument("--models", nargs="+", default=["linear", "lightgbm"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--max-train-rows", type=int, default=10000)
    ap.add_argument("--feats", nargs="+", choices=["all", "nodate"], default=["all", "nodate"])
    args = ap.parse_args()

    cfg = data_mod.load_config("configs/datasets.yaml")["datasets"][args.dataset]
    base = data_mod.build_task(data_mod.load_raw(cfg, "data/raw"), cfg, args.task)
    out = Path(f"results/label_delay/window_{args.dataset}_{args.task}.csv")
    for ratio in args.ratios:
        run_mod.label_available = windowed(ratio * args.gap_days)
        for feats in args.feats:
            task = copy.copy(base)
            drop = CALENDAR if feats == "nodate" else []
            task.X = base.X.drop(columns=drop)
            for policy in ["oracle", "known", "gap"]:
                meta = {"dataset": args.dataset, "drop_cols": " ".join(sorted(drop)),
                        "window_ratio": ratio, "window_days": ratio * args.gap_days}
                try:
                    df = run_mod.run_experiment(
                        task, "temporal", args.models, args.seeds, test_size=0.06, n_folds=10,
                        max_train_rows=args.max_train_rows, n_trials=0, meta=meta,
                        label_policy=policy, label_gap_days=args.gap_days,
                        train_sample="random", log=lambda *_: None)
                except ValueError as e:  # e.g. a short window of known labels holds one class
                    print(f"ratio={ratio} {feats} {policy}: skipped ({str(e)[:70]})", flush=True)
                    continue
                df.to_csv(out, mode="a", header=not out.exists(), index=False)
                print(f"ratio={ratio} {feats} {policy}: "
                      + df.groupby("model")["roc_auc"].mean().round(3).to_string().replace("\n", "  "),
                      flush=True)
    print("WINDOW DONE", flush=True)


if __name__ == "__main__":
    main()
