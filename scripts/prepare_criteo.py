"""Build the "conversion within N days of the click" task from the Criteo attribution data.

Source: Criteo Attribution Modeling for Bidding Dataset (Diemert, Meynet et al., AdKDD
2017), published by Criteo on Hugging Face (criteo/criteo-attribution-dataset),
CC BY-NC-SA 4.0. 16.5M ad impressions over 31 days; download it with
`bash scripts/download_data.sh criteo`.

Only clicked impressions are kept, and of those the users whose id falls in one of
--user-buckets buckets (default 50, i.e. 2%). The prediction is made at the click.

converted = 1 if a conversion followed within --days days of the click. A positive label is
known at the conversion, a negative one only when the window has passed (`deadline`), so
the outcome that is known early is the positive one: the opposite direction to Olist, where
the negative (on-time) label arrives first.

Caveats: timestamps in the file are seconds from the first impression, not real dates.
They are placed on an arbitrary origin (--origin), so the calendar features only encode the
position within the 31 days, and the weekday names are arbitrary (the 7-day cycle is real).
The official attribution window is 30 days; the N-day window is set here so that negative
labels can arrive inside the 31 days of data.

Usage: python scripts/prepare_criteo.py --days 7
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

CATS = ["campaign"] + [f"cat{i}" for i in range(1, 10)]
COLS = ["timestamp", "uid", "conversion", "conversion_timestamp", "click", "cost",
        "time_since_last_click"] + CATS


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--src", default="data/raw/criteo_attribution/criteo_attribution_dataset.tsv.gz")
    ap.add_argument("--out", default="data/raw/criteo_attribution.csv")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--user-buckets", type=int, default=50)
    ap.add_argument("--origin", default="2000-01-01", help="arbitrary date of timestamp 0")
    args = ap.parse_args()

    df = pd.read_csv(args.src, sep="\t", usecols=COLS)
    df = df[(df["click"] == 1) & (df["uid"] % args.user_buckets == 7)].drop(columns=["click"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    origin = pd.Timestamp(args.origin)
    delay_s = (df["conversion_timestamp"] - df["timestamp"]).where(df["conversion"] == 1)
    df["click_date"] = origin + pd.to_timedelta(df["timestamp"], unit="s")
    df["conversion_date"] = df["click_date"] + pd.to_timedelta(delay_s, unit="s")  # NaT if none
    df["deadline"] = df["click_date"] + pd.Timedelta(days=args.days)
    df["converted"] = (delay_s <= args.days * 86400).astype(int)
    df["click_hour"] = df["click_date"].dt.hour
    df["time_since_last_click"] = df["time_since_last_click"].where(df["time_since_last_click"] >= 0)
    for c in CATS:  # identifiers, not quantities: store as text so they are read as categories
        df[c] = "c" + df[c].astype(str)
    df = df.drop(columns=["timestamp", "conversion", "conversion_timestamp"])

    out = Path(args.out)
    df.to_csv(out, index=False)
    known_early = (df["conversion_date"] < df["deadline"]).mean()
    print(f"wrote {len(df)} clicks of {df['uid'].nunique()} users to {out}; "
          f"converted within {args.days} days: {df['converted'].mean():.3f} "
          f"(label known before the deadline for {known_early:.3f} of rows)")
    by_week = df.groupby((df["click_date"] - origin).dt.days // 7)["converted"].mean().round(3)
    print("rate by week of click:", by_week.to_dict())


if __name__ == "__main__":
    main()
