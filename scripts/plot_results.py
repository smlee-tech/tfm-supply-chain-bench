"""Plot the primary metric against training size, one line per model.

Usage: python scripts/plot_results.py results/dataco_late_delivery_curve.csv
"""
import sys

import matplotlib.pyplot as plt
import pandas as pd


def main(path: str) -> None:
    df = pd.read_csv(path)
    metric = "roc_auc" if df["task_type"].iloc[0] == "classification" else "rmse"
    # Rolling-origin folds have different training sizes when the cap is not reached,
    # so group by the requested cap and plot the mean size actually used.
    df["size"] = df["max_train_rows"].fillna(df["n_train"])
    agg = (df.groupby(["model", "size"])
             .agg(n_train=("n_train", "mean"), mean=(metric, "mean"), std=(metric, "std"))
             .reset_index())
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model, g in agg.groupby("model"):
        g = g.sort_values("n_train")
        ax.errorbar(g["n_train"], g["mean"], yerr=g["std"], marker="o", capsize=3, label=model)
    ax.set_xscale("log")
    ax.set_xlabel("training rows (log scale)")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{df['dataset'].iloc[0]} / {df['task'].iloc[0]} - temporal split")
    ax.grid(alpha=0.3)
    ax.legend()
    out = path.replace(".csv", ".png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main(sys.argv[1])
