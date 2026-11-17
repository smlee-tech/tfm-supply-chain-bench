"""Generate a synthetic order-line dataset with a known data-generating process.

This is a controlled testbed, not a substitute for real data: the mechanisms are chosen
by hand, so absolute scores say nothing about real supply chains. What it is good for is
checking that the benchmark measures what it claims to, because each effect has a knob:

  --drift 0        stationary control: grouped and temporal splits should score the same
  --drift 1        the world changes over time (see "Drift" below)
  --n-suppliers    cardinality of the supplier ID, which carries real signal (--keep-ids)
  --order-noise    unexplained delay per order; sets the ceiling (oracle AUC)
  --spike          short disruptions: every --spike-every days, delays rise for
                   --spike-length days, most for two carriers drawn anew each time
  --lead-scale     stretches promised and actual shipping times, and with them the time
                   until a label is known (on time: at shipping; late: at the promised date)

An order ships as one unit: all its lines share the delay and the label, as in DataCo.
A random split therefore puts lines with the same label on both sides (group leakage);
the grouped split removes that.

Drift (scaled by --drift):
  - covariate shift: express shipping and the region mix change over the three years;
  - concept shift: from the start of year 3 the carrier ranking reverses (the most
    reliable carrier becomes the least) and distance matters less. Learned relationships
    go stale while the task stays about as hard, so a drop is not a harder test set;
  - new categories: suppliers enter over time, so late test rows contain unseen IDs.

The file also holds oracle_p_late, the true probability of a late delivery given
everything except the noise. It is never a feature (drop_cols); the script prints the
AUC it reaches, which is the ceiling any model can approach.

Usage: python scripts/make_synthetic.py --n-orders 40000 --drift 1 --out data/raw/synthetic_orders.csv
"""
from __future__ import annotations

import argparse
from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

MODES = ["Standard", "Second Class", "First Class", "Same Day"]
SCHEDULED = np.array([5, 3, 2, 1])
REGIONS = ["North", "South", "East", "West", "Central", "Overseas"]
REGION_KM = np.array([300, 500, 700, 900, 400, 4000])
CARRIERS = ["Aster", "Borealis", "Cobalt", "Dune", "Ember"]
CARRIER_DELAY = np.array([-0.7, -0.3, 0.0, 0.3, 0.7])
CATEGORIES = ["apparel", "electronics", "furniture", "grocery", "tools", "toys", "pharma", "auto"]
CATEGORY_DELAY = np.array([0.0, 0.1, 0.6, -0.2, 0.2, 0.0, 0.3, 0.4])
_cdf = np.vectorize(lambda z: 0.5 * (1 + erf(z / sqrt(2))))


def generate(n_orders: int, drift: float, n_suppliers: int, order_noise: float,
             seed: int, spike: float = 0.0, spike_every: int = 120, spike_length: int = 60,
             lead_scale: float = 1.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rng_spike = np.random.default_rng(seed + 1)  # separate stream: spike=0 leaves the data as before
    days = 3 * 365
    order_day = np.sort(rng.integers(0, days, n_orders))
    u = order_day / days  # position on the timeline, 0..1
    late_era = (u >= 2 / 3).astype(float)

    # --- order level -------------------------------------------------------------
    p_mode = np.array([0.55, 0.2, 0.15, 0.10])[None, :] + drift * u[:, None] * np.array(
        [-0.25, 0.0, 0.10, 0.15])[None, :]
    mode = _choice_rows(rng, p_mode)
    p_region = np.array([0.2, 0.2, 0.2, 0.2, 0.15, 0.05])[None, :] + drift * u[:, None] * np.array(
        [-0.02, -0.02, -0.02, -0.02, -0.02, 0.10])[None, :]
    region = _choice_rows(rng, p_region)
    carrier = rng.integers(0, len(CARRIERS), n_orders)
    distance = REGION_KM[region] * rng.lognormal(0.0, 0.35, n_orders)
    n_lines = rng.choice([1, 2, 3, 4, 5], n_orders, p=[0.4, 0.25, 0.18, 0.1, 0.07])
    customer = rng.zipf(1.6, n_orders) % max(n_orders // 4, 1)
    order_eps = rng.normal(0, order_noise, n_orders)

    # --- line level --------------------------------------------------------------
    o = np.repeat(np.arange(n_orders), n_lines)
    n = len(o)
    category = rng.integers(0, len(CATEGORIES), n)
    quantity = rng.integers(1, 20, n)
    weight = np.round(rng.lognormal(1.0, 0.8, n) * (1 + 2 * (category == 2)), 2)
    unit_price = np.round(rng.lognormal(3.0, 0.9, n), 2)

    # Suppliers enter over time when drift > 0: supplier s exists from entry[s] on.
    supplier_effect = rng.normal(0, 0.6, n_suppliers)
    entry = np.where(rng.random(n_suppliers) < 0.5, 0.0, rng.random(n_suppliers) * drift)
    supplier = np.empty(n, dtype=int)
    line_u = u[o]
    order_by_entry = np.argsort(entry)
    n_available = np.searchsorted(entry[order_by_entry], line_u, side="right")
    supplier = order_by_entry[(rng.random(n) * n_available).astype(int)]

    # --- delay model -------------------------------------------------------------
    season = 0.5 * np.sin(2 * np.pi * (order_day[o] % 365) / 365 - 1.2)  # peak late in the year
    weekend = np.isin(order_day[o] % 7, [5, 6]) * 0.3
    dist_slope = 0.9 * (1 - 0.4 * drift * late_era[o])
    w = drift * late_era[o]  # carrier ranking reverses in year 3
    carrier_delay = (1 - w) * CARRIER_DELAY[carrier[o]] + w * CARRIER_DELAY[::-1][carrier[o]]
    # Disruptions: the last spike_length days of every spike_every-day period. All orders
    # are delayed a little and two carriers a lot; which two changes each time, so a
    # disruption cannot be learned from the previous ones.
    period = order_day // spike_every
    in_spike = (order_day % spike_every) >= (spike_every - spike_length)
    hit = np.zeros((period.max() + 1, len(CARRIERS)))
    for k in range(len(hit)):
        hit[k, rng_spike.choice(len(CARRIERS), 2, replace=False)] = 1.0
    spike_delay = spike * in_spike * (0.5 + hit[period, carrier])
    line_mean = (
        -0.2
        + spike_delay[o]
        + dist_slope * np.log(distance[o] / 600)
        + carrier_delay
        + CATEGORY_DELAY[category]
        + supplier_effect[supplier]
        + season + weekend
        + 0.25 * np.log(weight) * (mode[o] >= 2)  # heavy items hurt express shipping
        + 0.015 * quantity
    )
    # One shipment per order: the order's delay is the mean over its lines plus order noise.
    mean_extra = (np.bincount(o, weights=line_mean) / n_lines)[o]
    extra = mean_extra + order_eps[o]
    scheduled = SCHEDULED[mode[o]]
    ship_days = np.maximum(0, np.round(scheduled + extra)).astype(int)
    # late <=> round(scheduled + extra) > scheduled <=> extra >= 0.5
    oracle = _cdf((mean_extra - 0.5) / order_noise)

    # Stretch the calendar, not the label: late is decided before scaling.
    late = (ship_days > scheduled).astype(int)
    scheduled = np.round(lead_scale * scheduled).astype(int)
    ship_days = np.round(lead_scale * ship_days).astype(int)
    order_date = pd.Timestamp("2022-01-01") + pd.to_timedelta(order_day[o], unit="D")
    return pd.DataFrame({
        "order_id": np.char.add("O", o.astype(str)),
        "order_date": order_date.strftime("%Y-%m-%d"),
        "customer_id": customer[o],
        "supplier_id": np.char.add("S", supplier.astype(str)),
        "shipping_mode": np.array(MODES)[mode[o]],
        "scheduled_days": scheduled,
        "region": np.array(REGIONS)[region[o]],
        "carrier": np.array(CARRIERS)[carrier[o]],
        "distance_km": np.round(distance[o], 1),
        "category": np.array(CATEGORIES)[category],
        "quantity": quantity,
        "weight_kg": weight,
        "unit_price": unit_price,
        "promised_date": (order_date + pd.to_timedelta(scheduled, unit="D")).strftime("%Y-%m-%d"),
        "ship_date": (order_date + pd.to_timedelta(ship_days, unit="D")).strftime("%Y-%m-%d"),
        "ship_days": ship_days,
        "late": late,
        "oracle_p_late": np.round(oracle, 4),
    })


def _choice_rows(rng: np.random.Generator, p: np.ndarray) -> np.ndarray:
    """One categorical draw per row of a probability matrix."""
    p = np.clip(p, 1e-6, None)
    cdf = np.cumsum(p / p.sum(axis=1, keepdims=True), axis=1)
    return (rng.random(len(p))[:, None] > cdf).sum(axis=1).clip(max=p.shape[1] - 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--n-orders", type=int, default=40_000)
    ap.add_argument("--drift", type=float, default=1.0, help="0 = stationary, 1 = default drift")
    ap.add_argument("--n-suppliers", type=int, default=2_000)
    ap.add_argument("--order-noise", type=float, default=0.8)
    ap.add_argument("--spike", type=float, default=0.0, help="0 = no disruptions")
    ap.add_argument("--spike-every", type=int, default=120)
    ap.add_argument("--spike-length", type=int, default=60)
    ap.add_argument("--lead-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/raw/synthetic_orders.csv")
    args = ap.parse_args()

    df = generate(args.n_orders, args.drift, args.n_suppliers, args.order_noise, args.seed,
                  args.spike, args.spike_every, args.spike_length, args.lead_scale)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    from sklearn.metrics import roc_auc_score

    year = pd.to_datetime(df["order_date"]).dt.year
    print(f"wrote {len(df)} order lines ({df['order_id'].nunique()} orders) to {out}")
    for y, g in df.groupby(year):
        print(f"  {y}: late rate {g['late'].mean():.3f}, "
              f"oracle AUC {roc_auc_score(g['late'], g['oracle_p_late']):.3f}, "
              f"suppliers {g['supplier_id'].nunique()}")


if __name__ == "__main__":
    main()
