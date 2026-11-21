"""Tables and figures for the label-delay study, straight from the results CSVs.

Usage:
  python scripts/summarize_label_delay.py results/label_delay/untuned_*.csv \
      results/label_delay/tuned_*.csv [--plot docs/label_delay.png]
  python scripts/summarize_label_delay.py results/label_delay/window_*.csv \
      [--plot-window docs/context_window.png]
  python scripts/summarize_label_delay.py results/label_delay/prior_*.csv

Grid CSVs (untuned_*, tuned_*) hold one dataset/task run with --split temporal over
train sample (recent, random) x label policy (oracle, known, gap) x feature set. Window
CSVs (window_*, from scripts/context_window.py) add the length of the training window as a
multiple of the label delay. Prior CSVs (prior_*, from scripts/prior_correction.py) hold
the prior-shift correction of `known`. The kinds are told apart by their columns
(`variant` for prior CSVs, `window_ratio` for window CSVs).

Feature sets: all; "no month/day-of-year" (without pred_month, pred_dayofyear); "no
calendar" (also without pred_dayofweek).

Scores are first averaged over seeds per cutoff; differences are paired per cutoff, with a
bootstrap 95% interval over cutoffs. Cutoffs are consecutive and their training windows
overlap, so they are not independent and the intervals are a lower bound on the real
uncertainty.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

POLICIES = ["oracle", "known", "gap"]
MODELS = ["tabpfn", "lightgbm", "linear"]
FEATURES = ["all", "no month/day-of-year", "no calendar"]
COLORS = {"oracle": "#4c4c4c", "known": "#d1495b", "gap": "#2e86ab"}


def load(path: str, seeds: list[int]) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "variant" in df.columns:  # prior-correction CSV: always without calendar features
        return df[df["seed"].isin(seeds)]
    drop = df["drop_cols"].fillna("")
    df["features"] = np.where(drop == "", FEATURES[0],
                              np.where(drop.str.contains("pred_dayofweek"), FEATURES[2],
                                       FEATURES[1]))
    return df[df["seed"].isin(seeds)]


def per_cutoff(df: pd.DataFrame, keys: list[str], metric: str = "roc_auc") -> pd.DataFrame:
    """Mean score over seeds: one row per keys + cutoff, one column per label policy."""
    out = df.groupby(keys + ["fold", "label_policy"])[metric].mean().unstack("label_policy")
    return out.reindex(columns=POLICIES)


def paired(diff: np.ndarray, rng: np.random.Generator) -> str:
    diff = diff[~np.isnan(diff)]
    if len(diff) == 0:
        return "n/a"
    boot = rng.choice(diff, size=(10_000, len(diff))).mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return f"{diff.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]"


def seeds_by_model(df: pd.DataFrame) -> str:
    n = df.groupby("model")["seed"].nunique()
    return ", ".join(f"{m}: {n[m]}" for m in MODELS if m in n.index)


def ordered(index_values, order):
    return [v for v in order if v in set(index_values)]


def tables(df: pd.DataFrame) -> None:
    name = f"{df['dataset'].iloc[0]} / {df['task'].iloc[0]}"
    trials = "tuned" if df["tuned_params"].notna().any() else "untuned"
    print(f"\n### {name} ({df['fold'].nunique()} cutoffs, seeds per model: {seeds_by_model(df)}; "
          f"{int(df['max_train_rows'].iloc[0]):,} training rows, {trials})\n")
    keys = ["model", "features", "train_sample"]
    auc, ll = per_cutoff(df, keys), per_cutoff(df, keys, "log_loss")
    rng = np.random.default_rng(0)
    models = ordered(auc.index.get_level_values("model"), MODELS)

    print("Mean ROC-AUC over cutoffs (worst cutoff in brackets), then mean log loss:\n")
    print("| Model | Features | Train sample | oracle | known | gap | log loss oracle / known / gap |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for model in models:
        for features in ordered(auc.loc[model].index.get_level_values("features"), FEATURES):
            for sample, g in auc.loc[(model, features)].groupby(level="train_sample"):
                cells = " | ".join(f"{g[p].mean():.3f} ({g[p].min():.3f})" for p in POLICIES)
                losses = " / ".join(f"{ll.loc[(model, features, sample), p].mean():.3f}"
                                    for p in POLICIES)
                print(f"| {model} | {features} | {sample} | {cells} | {losses} |")

    print("\nPaired ROC-AUC differences per cutoff, recent training sample "
          "(mean [bootstrap 95% interval over cutoffs]):\n")
    print("| Model | Features | known - oracle | gap - oracle | known - gap |")
    print("| --- | --- | --- | --- | --- |")
    for model in models:
        for features in ordered(auc.loc[model].index.get_level_values("features"), FEATURES):
            if (model, features, "recent") not in auc.index.droplevel("fold"):
                continue
            g = auc.loc[(model, features, "recent")]
            cells = [paired((g[a] - g[b]).to_numpy(), rng)
                     for a, b in [("known", "oracle"), ("gap", "oracle"), ("known", "gap")]]
            print(f"| {model} | {features} | " + " | ".join(cells) + " |")


def ratio_label(r: float) -> str:
    return "all history" if np.isinf(r) else f"{r:g}x"


def window_tables(df: pd.DataFrame) -> None:
    name = f"{df['dataset'].iloc[0]} / {df['task'].iloc[0]}"
    print(f"\n### Context window: {name} ({df['fold'].nunique()} cutoffs, seeds per model: "
          f"{seeds_by_model(df)}; at most {int(df['max_train_rows'].iloc[0]):,} training rows, "
          f"untuned)\n")
    keys = ["model", "features", "window_ratio"]
    auc, ll = per_cutoff(df, keys), per_cutoff(df, keys, "log_loss")
    rows = df.groupby(keys)["n_train"].mean()
    rng = np.random.default_rng(0)

    print("Window = multiple of the label delay. Mean over cutoffs; a dash means the policy "
          "could not run (one class in the window).\n")
    print("| Model | Features | Window | ROC-AUC oracle / known / gap | log loss oracle / known / gap "
          "| known - gap, ROC-AUC | known - gap, log loss | mean training rows |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")

    def fmt(g):
        return " / ".join("–" if g[p].isna().all() else f"{g[p].mean():.3f}" for p in POLICIES)

    for model in ordered(auc.index.get_level_values("model"), MODELS):
        for features in ordered(auc.loc[model].index.get_level_values("features"), FEATURES):
            for ratio in sorted(auc.loc[(model, features)].index.get_level_values("window_ratio")
                                .unique()):
                a, b = auc.loc[(model, features, ratio)], ll.loc[(model, features, ratio)]
                print(f"| {model} | {features} | {ratio_label(ratio)} | {fmt(a)} | {fmt(b)} | "
                      f"{paired((a['known'] - a['gap']).to_numpy(), rng)} | "
                      f"{paired((b['known'] - b['gap']).to_numpy(), rng)} | "
                      f"{rows.loc[(model, features, ratio)]:,.0f} |")


def prior_tables(df: pd.DataFrame) -> None:
    name = f"{df['dataset'].iloc[0]} / {df['task'].iloc[0]}"
    print(f"\n### Prior-shift correction: {name} ({df['fold'].nunique()} cutoffs, seeds per "
          f"model: {seeds_by_model(df)}; no calendar features, untuned)\n")
    variants = ["oracle", "known", "gap", "known+mature", "known+oracle"]
    ll = (df.groupby(["model", "window_ratio", "fold", "variant"])["log_loss"].mean()
          .unstack("variant")[variants])
    ece = df.groupby(["model", "window_ratio", "variant"])["ece"].mean().unstack("variant")
    rng = np.random.default_rng(0)
    print("known+mature: prior from the rows old enough to be fully labelled; known+oracle: "
          "true prior of the window (not usable in practice). ROC-AUC is unchanged by the "
          "correction.\n")
    print("| Model | Window | log loss oracle / known / gap / known+mature / known+oracle "
          "| known+mature - gap | known+mature - known | ECE known / gap / known+mature |")
    print("| --- | --- | --- | --- | --- | --- |")
    for model in ordered(ll.index.get_level_values("model"), MODELS):
        for ratio in sorted(ll.loc[model].index.get_level_values("window_ratio").unique()):
            g, e = ll.loc[(model, ratio)], ece.loc[(model, ratio)]
            print(f"| {model} | {ratio_label(ratio)} | "
                  + " / ".join(f"{g[v].mean():.3f}" for v in variants) + " | "
                  + paired((g["known+mature"] - g["gap"]).to_numpy(), rng) + " | "
                  + paired((g["known+mature"] - g["known"]).to_numpy(), rng) + " | "
                  + " / ".join(f"{e[v]:.3f}" for v in ["known", "gap", "known+mature"]) + " |")


def plot(frames: list[pd.DataFrame], out: str) -> None:
    import matplotlib.pyplot as plt

    frames = [df for df in frames if (df["model"] == "tabpfn").any()
              and df["tuned_params"].isna().all()]
    fig, axes = plt.subplots(len(frames), 2, figsize=(11, 3.6 * len(frames)), sharey="row",
                             squeeze=False)
    for row, df in zip(axes, frames):
        pc = per_cutoff(df[df["model"] == "tabpfn"], ["model", "features", "train_sample"])
        cutoffs = pd.to_datetime(df.groupby("fold")["cutoff"].first()).dt.strftime("%Y-%m-%d")
        for ax, features in zip(row, FEATURES[:2]):
            g = pc.loc[("tabpfn", features, "recent")]
            for policy in POLICIES:
                ax.plot(cutoffs.to_numpy(), g[policy].to_numpy(), marker="o", ms=4,
                        color=COLORS[policy], label=policy)
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


def plot_window(frames: list[pd.DataFrame], out: str) -> None:
    """ROC-AUC against window length, no calendar features: TabPFN solid, LightGBM dashed."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(frames), figsize=(4.6 * len(frames), 3.8), squeeze=False)
    for ax, df in zip(axes[0], frames):
        pc = per_cutoff(df[df["features"] == FEATURES[2]], ["model", "window_ratio"])
        mean = pc.groupby(level=["model", "window_ratio"]).mean()
        ratios = sorted(r for r in mean.loc["tabpfn"].index if r >= 1)
        x = np.arange(len(ratios))
        for model, style in [("tabpfn", "-"), ("lightgbm", "--")]:
            for policy in POLICIES:
                y = mean.loc[model].reindex(ratios)[policy].to_numpy()
                ax.plot(x, y, style, marker="o", ms=4, color=COLORS[policy],
                        alpha=1.0 if model == "tabpfn" else 0.55,
                        label=f"{policy}, {'TabPFN-3.5' if model == 'tabpfn' else 'LightGBM'}")
        ax.set_xticks(x, [ratio_label(r) for r in ratios], fontsize=8)
        ax.set_title(df["dataset"].iloc[0], fontsize=10)
        ax.set_xlabel("training window / label delay")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("ROC-AUC")
    axes[0][0].legend(fontsize=7)
    fig.suptitle("Length of the training window, no calendar features (mean over 10 cutoffs)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"\nsaved {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2],
                    help="seeds to include (default 0 1 2, the set every model has)")
    ap.add_argument("--plot", help="write the per-cutoff TabPFN figure to this PNG")
    ap.add_argument("--plot-window", help="write the window-length figure to this PNG")
    args = ap.parse_args()
    frames = [load(p, args.seeds) for p in args.paths]
    priors = [df for df in frames if "variant" in df.columns]
    rest = [df for df in frames if "variant" not in df.columns]
    grids = [df for df in rest if "window_ratio" not in df.columns]
    windows = [df for df in rest if "window_ratio" in df.columns]
    for df in grids:
        tables(df)
    for df in windows:
        window_tables(df)
    for df in priors:
        prior_tables(df)
    if args.plot:
        plot(grids, args.plot)
    if args.plot_window:
        plot_window(windows, args.plot_window)


if __name__ == "__main__":
    main()
