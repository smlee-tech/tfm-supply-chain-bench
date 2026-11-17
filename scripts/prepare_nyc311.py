"""Download a sample of NYC 311 service requests and build the "slow request" task table.

Source: NYC Open Data, "311 Service Requests from 2020 to Present" (dataset erm2-nwe9),
read through the Socrata API. One agency is kept, because closing times differ by orders of
magnitude between agencies (police requests close within minutes). The sample is every
request whose key ends in --key-suffix (one digit = 10%).

slow = 1 if the request was not closed within --days days of being filed. A fast label is
known when the request is closed, a slow label when the deadline passes; `deadline` holds
that date for the label-delay policies. Caveat: many requests are closed in batches (same
second, or exactly midnight), so closing times partly reflect administration.

Usage: python scripts/prepare_nyc311.py --agency HPD --start 2020-01-01 --end 2021-12-31
"""
from __future__ import annotations

import argparse
import urllib.parse
from pathlib import Path

import pandas as pd

URL = "https://data.cityofnewyork.us/resource/erm2-nwe9.csv"
COLS = ["unique_key", "created_date", "closed_date", "complaint_type", "descriptor",
        "location_type", "incident_zip", "borough", "open_data_channel_type"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--agency", default="HPD")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2021-12-31")
    ap.add_argument("--key-suffix", default="7")
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--out", default=None, help="default: data/raw/nyc311_<agency>.csv")
    args = ap.parse_args()

    where = (f"created_date between '{args.start}T00:00:00' and '{args.end}T23:59:59' "
             f"AND agency='{args.agency}' AND unique_key like '%{args.key_suffix}'")
    query = urllib.parse.urlencode({"$select": ",".join(COLS), "$where": where,
                                    "$limit": 5_000_000})
    df = pd.read_csv(f"{URL}?{query}", parse_dates=["created_date", "closed_date"],
                     dtype={"incident_zip": str, "unique_key": str})

    lag_days = (df["closed_date"] - df["created_date"]).dt.total_seconds() / 86400
    df = df[~(lag_days < 0)].sort_values("created_date").reset_index(drop=True)  # bad records
    lag_days = (df["closed_date"] - df["created_date"]).dt.total_seconds() / 86400
    df["slow"] = (~(lag_days <= args.days)).astype(int)  # never closed counts as slow
    df["deadline"] = df["created_date"] + pd.Timedelta(days=args.days)
    df["zip3"] = df["incident_zip"].str[:3]
    df["created_hour"] = df["created_date"].dt.hour
    df = df.drop(columns=["incident_zip"])

    out = Path(args.out or f"data/raw/nyc311_{args.agency.lower()}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    by_quarter = df.groupby(df["created_date"].dt.to_period("Q"))["slow"].mean().round(3)
    print(f"wrote {len(df)} requests to {out}; slow rate {df['slow'].mean():.3f}")
    print("slow rate by quarter:", by_quarter.to_dict())


if __name__ == "__main__":
    main()
