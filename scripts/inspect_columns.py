"""Check that the columns named in configs/datasets.yaml exist in the downloaded files.

Usage: python scripts/inspect_columns.py dataco
"""
import sys

import pandas as pd

sys.path.insert(0, "src")
from tfmbench.data import load_config, load_raw  # noqa: E402


def main(name: str) -> int:
    ds = load_config("configs/datasets.yaml")["datasets"][name]
    df = load_raw(ds, "data/raw")
    print(f"{name}: {len(df)} rows, {df.shape[1]} columns\n")
    print(pd.DataFrame({"dtype": df.dtypes.astype(str),
                        "null_rate": df.isna().mean().round(3)}).to_string())

    named = {ds["time_col"], *ds.get("drop_cols", []), *ds.get("id_cols", []),
             *ds.get("other_date_cols", [])}
    for task in ds["tasks"].values():
        named |= set(task.get("leakage_cols", []))
        named |= {v for k, v in task["target"].items() if k != "kind"}
    missing = sorted(c for c in named if c not in df.columns)
    print("\nColumns named in config but missing from the file:", missing or "none")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
