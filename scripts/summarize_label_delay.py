"""Tables and a figure for the label-delay study, straight from the results CSVs.

Usage: python scripts/summarize_label_delay.py results/label_delay/untuned_*.csv [--plot docs/label_delay.png]

Each CSV holds one dataset/task run with --split temporal over the grid
train sample (recent, random) x label policy (oracle, known, gap) x features
(all, without pred_month and pred_dayofyear). Scores are first averaged over seeds per
cutoff; differences are paired per cutoff, with a bootstrap 95% interval over cutoffs.
Cutoffs are consecutive and their training windows overlap, so they are not independent
and the intervals are a lower bound on the real uncertainty.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

POLICIES = ["oracle", "known", "gap"]
MODELS = ["tabpfn", "lightgbm", "linear"]


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["features"] = np.where(df["drop_cols"].isna(), "all", "no month/day-of-year")
    return df


def per_cutoff(df: pd.DataFrame) -> pd.DataFrame:
    """Mean ROC-AUC over seeds: one row per model, features, sample, cutoff; one column per policy."""
    keys = ["model", "features", "train_sample", "fold", "label_policy"]
    return df.groupby(keys)["roc_auc"].mean().unstack("label_policy")[POLICIES]


def paired(diff: np.ndarray, rng: np.random.Generator) -> str:
    boot = rng.choice(diff, size=(10_000, len(diff))).mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return f"{diff.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]"


def tables(df: pd.DataFrame) -> None:
    name = f"{df['dataset'].iloc[0]} / {df['task'].iloc[0]}"
    n_seeds, n_folds = df["seed"].nunique(), df["fold"].nunique()
    trials = "tuned" if df["tuned_params"].notna().any() else "untuned"
    print(f"\n### {name} ({n_folds} cutoffs x {n_seeds} seeds, "
          f"{int(df['max_train_rows'].iloc[0]):,} training rows, {trials})\n")
    pc = per_cutoff(df)
    rng = np.random.default_rng(0)

    print("Mean ROC-AUC over cutoffs (worst cutoff in brackets):\n")
    print("| Model | Features | Train sample | oracle | known | gap |")
    print("| --- | --- | --- | --- | --- | --- |")
    for model in [m for m in MODELS if m in pc.index.get_level_values("model")]:
        for (features, sample), g in pc.loc[model].groupby(level=["features", "train_sample"]):
            cells = " | ".join(f"{g[p].mean():.3f} ({g[p].min():.3f})" for p in POLICIES)
            print(f"| {model} | {features} | {sample} | {cells} |")

    print("\nPaired differences per cutoff, recent training sample "
          "(mean [bootstrap 95% interval over cutoffs]):\n")
    print("| Model | known - oracle, all features | known - oracle, no month/day-of-year "
          "| gap - oracle, all features | dropping month/day-of-year under known |")
    print("| --- | --- | --- | --- | --- |")
    for model in [m for m in MODELS if m in pc.index.get_level_values("model")]:
        a = pc.loc[(model, "all", "recent")]
        b = pc.loc[(model, "no month/day-of-year", "recent")]
        cells = [paired((a["known"] - a["oracle"]).to_numpy(), rng),
                 paired((b["known"] - b["oracle"]).to_numpy(), rng),
                 paired((a["gap"] - a["oracle"]).to_numpy(), rng),
                 paired((b["known"] - a["known"]).to_numpy(), rng)]
        print(f"| {model} | " + " | ".join(cells) + " |")


def plot(frames: list[pd.DataFrame], out: str) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(frames), 2, figsize=(11, 3.6 * len(frames)), sharey="row",
                             squeeze=False)
    colors = {"oracle": "#4c4c4c", "known": "#d1495b", "gap": "#2e86ab"}
    for row, df in zip(axes, frames):
        pc = per_cutoff(df[df["model"] == "tabpfn"])
        cutoffs = pd.to_datetime(df.groupby("fold")["cutoff"].first()).dt.strftime("%Y-%m-%d")
        for ax, features in zip(row, ["all", "no month/day-of-year"]):
            g = pc.loc[("tabpfn", features, "recent")]
            for policy in POLICIES:
                ax.plot(cutoffs.to_numpy(), g[policy].to_numpy(), marker="o", ms=4,
                        color=colors[policy], label=policy)
            ax.set_title(f"{df['dataset'].iloc[0]}: TabPFN-3.5, features: {features}", fontsize=10)
            ax.tick_params(axis="x", rotation=60, labelsize=7)
            ax.grid(alpha=0.3)
        row[0].set_ylabel("ROC-AUC")
    axes[0][0].legend(title="label policy", fontsize=8)
    fig.suptitle("Context of the 10,000 most recent rows, by label policy and test cutoff",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"\nsaved {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--plot", help="write the per-cutoff TabPFN figure to this PNG")
    args = ap.parse_args()
    frames = [load(p) for p in args.paths]
    for df in frames:
        tables(df)
    if args.plot:
        plot(frames, args.plot)


if __name__ == "__main__":
    main()
