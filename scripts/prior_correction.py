"""Check experiment: does a prior-shift correction repair the probabilities of `known`?

Training on the recent rows whose label has arrived (`known`) over-represents the outcome
that is labelled first, so predicted probabilities are shifted even when the ranking is
fine. If the features do not tell how old a row is (no calendar features), the selection
depends on the outcome only, which is label shift, and the standard correction applies:

    odds' = odds * (pi / (1 - pi)) / (pi_train / (1 - pi_train))

pi_train is the positive rate of the `known` training rows. Two estimates of pi:
- mature: the positive rate of the rows old enough for every label to have arrived (the
  `gap` rows of the same window length). Usable in practice, but older.
- oracle: the true positive rate of all rows in the window. Not usable in practice; it
  shows what is left when the prior is right.

The correction is monotone, so ROC-AUC is unchanged; log loss, Brier and ECE move.
Variants per cutoff: oracle, known, gap, known+mature, known+oracle. Windows are multiples
of the label delay, as in scripts/context_window.py. No calendar features.

Usage: python scripts/prior_correction.py --dataset olist --task late_delivery --gap-days 30
       (run TabPFN in its own call: --models tabpfn --seeds 0)
Results append to results/label_delay/prior_<dataset>_<task>.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from tfmbench import data as data_mod
from tfmbench.metrics import classification_metrics
from tfmbench.models import REGISTRY, predict_scores
from tfmbench.preprocess import categorical_indices, make_preprocessor, to_float
from tfmbench.splits import label_available, make_folds, subsample

CALENDAR = ["pred_month", "pred_dayofyear", "pred_dayofweek"]


def window(idx: np.ndarray, t: pd.Series, days: float) -> np.ndarray:
    """The last `days` before the latest row of idx."""
    if np.isinf(days):
        return idx
    tt = pd.Series(pd.to_datetime(t)).iloc[idx]
    return idx[(tt >= tt.max() - pd.Timedelta(days=days)).to_numpy()]


def shift_prior(p: np.ndarray, pi_train: float, pi: float) -> np.ndarray:
    odds = p / (1 - p) * (pi / (1 - pi)) / (pi_train / (1 - pi_train))
    return odds / (1 + odds)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--gap-days", type=float, required=True, help="label delay G of the task")
    ap.add_argument("--ratios", nargs="+", type=float, default=[1.25, 2, 3, 4, np.inf])
    ap.add_argument("--models", nargs="+", default=["linear", "lightgbm"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--max-train-rows", type=int, default=10000)
    args = ap.parse_args()

    cfg = data_mod.load_config("configs/datasets.yaml")["datasets"][args.dataset]
    task = data_mod.build_task(data_mod.load_raw(cfg, "data/raw"), cfg, args.task)
    X, y, t = task.X.drop(columns=CALENDAR), task.y.to_numpy(), task.t
    out = Path(f"results/label_delay/prior_{args.dataset}_{args.task}.csv")
    rows = []
    for ratio in args.ratios:
        days = ratio * args.gap_days
        for seed in args.seeds:
            for fold in make_folds("temporal", t, task.groups, 0.06, seed, 10):
                usable = {p: window(label_available(fold.train, t, task.label_t, fold.cutoff, p,
                                                    args.gap_days), t, days)
                          for p in ["oracle", "known", "gap"]}
                priors = {"mature": float(y[usable["gap"]].mean()),
                          "oracle": float(y[usable["oracle"]].mean())}
                y_te = y[fold.test]
                for name in args.models:
                    spec = REGISTRY[name]
                    cat_idx = categorical_indices(X) if spec.family == "tree" else []
                    for policy, idx in usable.items():
                        train_idx = subsample(idx, args.max_train_rows, seed, "random", t)
                        pi_train = float(y[train_idx].mean())
                        pre = make_preprocessor(spec.family, X.iloc[train_idx])
                        model = spec.build(task.task_type, seed, {}, cat_idx)
                        model.fit(to_float(pre.fit_transform(X.iloc[train_idx])), y[train_idx])
                        p = predict_scores(model, to_float(pre.transform(X.iloc[fold.test])),
                                           task.task_type)
                        variants = {policy: (p, None)}
                        if policy == "known":
                            pc = np.clip(p, 1e-7, 1 - 1e-7)
                            for kind, pi in priors.items():
                                variants[f"known+{kind}"] = (shift_prior(pc, pi_train, pi), pi)
                        for variant, (scores, pi) in variants.items():
                            rows.append({
                                "dataset": args.dataset, "task": task.name, "model": name,
                                "seed": seed, "fold": fold.fold, "cutoff": fold.cutoff,
                                "window_ratio": ratio, "window_days": days, "variant": variant,
                                "n_train": len(train_idx),
                                "train_target_mean": round(pi_train, 5),
                                "prior_used": None if pi is None else round(pi, 5),
                                "test_target_mean": round(float(y_te.mean()), 5),
                                "pred_mean": round(float(np.mean(scores)), 5),
                                **classification_metrics(y_te, scores),
                            })
        df = pd.DataFrame(rows)
        df.to_csv(out, mode="a", header=not out.exists(), index=False)
        rows = []
        print(f"ratio={ratio}: " + df.groupby(["model", "variant"])["log_loss"].mean().round(3)
              .to_string().replace("\n", "  "), flush=True)
    print("PRIOR DONE", flush=True)


if __name__ == "__main__":
    main()
