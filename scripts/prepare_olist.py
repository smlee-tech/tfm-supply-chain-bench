"""Join the Olist tables into one row per delivered order: data/raw/olist_orders.csv.

Prediction time is the moment of purchase. Features are things known then: what was
bought, from which seller, to where, the price and freight paid, the payment, and the
delivery date promised to the customer. Everything recorded later (approval, hand-over
to the carrier, delivery, reviews) is either excluded or kept only as a leakage column.

Targets:
  late           1 if the order arrived on a later calendar day than the promised date
  delivery_days  days from purchase to delivery

Only orders with status "delivered" and a delivery date have a label; the rest
(about 3%: cancelled, unavailable, still in transit) are dropped.

Usage: python scripts/prepare_olist.py [--src data/raw/olist] [--out data/raw/olist_orders.csv]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _first(s: pd.Series):
    return s.iloc[0]


def _haversine_km(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def build(src: Path) -> pd.DataFrame:
    def read(name, **kw):
        return pd.read_csv(src / name, **kw)

    zips = {"dtype": str}
    orders = read("olist_orders_dataset.csv")
    items = read("olist_order_items_dataset.csv")
    customers = read("olist_customers_dataset.csv", dtype={"customer_zip_code_prefix": str})
    sellers = read("olist_sellers_dataset.csv", dtype={"seller_zip_code_prefix": str})
    products = read("olist_products_dataset.csv")
    payments = read("olist_order_payments_dataset.csv")
    geo = read("olist_geolocation_dataset.csv", dtype={"geolocation_zip_code_prefix": str})
    names = read("product_category_name_translation.csv", encoding="utf-8-sig")
    del zips

    date_cols = ["order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
                 "order_delivered_customer_date", "order_estimated_delivery_date"]
    for c in date_cols:
        orders[c] = pd.to_datetime(orders[c], errors="coerce")
    orders = orders[(orders["order_status"] == "delivered")
                    & orders["order_delivered_customer_date"].notna()].copy()

    # --- items: one row per order, described by its first item plus totals -----------
    products = products.merge(names, on="product_category_name", how="left")
    products["category"] = products["product_category_name_english"].fillna(
        products["product_category_name"])
    products["volume_cm3"] = (products["product_length_cm"] * products["product_height_cm"]
                              * products["product_width_cm"])
    items = items.sort_values(["order_id", "order_item_id"]).merge(
        products[["product_id", "category", "product_weight_g", "volume_cm3"]],
        on="product_id", how="left")
    items["shipping_limit_date"] = pd.to_datetime(items["shipping_limit_date"], errors="coerce")
    per_order = items.groupby("order_id").agg(
        n_items=("order_item_id", "size"),
        n_products=("product_id", "nunique"),
        n_sellers=("seller_id", "nunique"),
        seller_id=("seller_id", _first),
        category=("category", _first),
        price_total=("price", "sum"),
        price_max=("price", "max"),
        freight_total=("freight_value", "sum"),
        weight_total_g=("product_weight_g", "sum"),
        weight_max_g=("product_weight_g", "max"),
        volume_total_cm3=("volume_cm3", "sum"),
        shipping_limit=("shipping_limit_date", "max"),
    ).reset_index()

    # --- payments ------------------------------------------------------------------
    payments = payments.sort_values(["order_id", "payment_value"], ascending=[True, False])
    pay = payments.groupby("order_id").agg(
        payment_type=("payment_type", _first),
        payment_installments=("payment_installments", "max"),
        payment_value=("payment_value", "sum"),
        n_payments=("payment_sequential", "size"),
    ).reset_index()

    # --- geography: zip-prefix centroid (median is robust to the bad points in the file) --
    centroid = geo.groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]].median()

    df = (orders.merge(customers, on="customer_id", how="left")
                .merge(per_order, on="order_id", how="inner")
                .merge(sellers, on="seller_id", how="left")
                .merge(pay, on="order_id", how="left"))
    c = centroid.reindex(df["customer_zip_code_prefix"]).to_numpy()
    s = centroid.reindex(df["seller_zip_code_prefix"]).to_numpy()
    df["customer_lat"], df["customer_lng"] = c[:, 0], c[:, 1]
    df["distance_km"] = np.round(_haversine_km(s[:, 0], s[:, 1], c[:, 0], c[:, 1]), 1)
    df["same_state"] = (df["customer_state"] == df["seller_state"]).astype(int)

    purchase = df["order_purchase_timestamp"]
    day = pd.Timedelta(days=1)
    df["promised_days"] = (df["order_estimated_delivery_date"] - purchase) / day
    df["shipping_limit_days"] = (df["shipping_limit"] - purchase) / day
    df["purchase_hour"] = purchase.dt.hour
    df["route"] = df["seller_state"] + ">" + df["customer_state"]

    delivered = df["order_delivered_customer_date"]
    df["delivery_days"] = (delivered - purchase) / day
    df["late"] = (delivered.dt.normalize() > df["order_estimated_delivery_date"].dt.normalize()).astype(int)

    cols = [
        "order_id", "customer_unique_id", "order_purchase_timestamp", "purchase_hour",
        "promised_days", "shipping_limit_days",
        "customer_zip_code_prefix", "customer_city", "customer_state", "customer_lat", "customer_lng",
        "seller_id", "seller_zip_code_prefix", "seller_city", "seller_state",
        "route", "same_state", "distance_km",
        "n_items", "n_products", "n_sellers", "category",
        "price_total", "price_max", "freight_total",
        "weight_total_g", "weight_max_g", "volume_total_cm3",
        "payment_type", "payment_installments", "payment_value", "n_payments",
        # recorded after the purchase: leakage columns and targets
        "order_approved_at", "order_delivered_carrier_date", "order_delivered_customer_date",
        "order_estimated_delivery_date", "delivery_days", "late",
    ]
    return df[cols].sort_values("order_purchase_timestamp").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--src", default="data/raw/olist")
    ap.add_argument("--out", default="data/raw/olist_orders.csv")
    args = ap.parse_args()
    df = build(Path(args.src))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    t = df["order_purchase_timestamp"]
    print(f"wrote {len(df)} orders to {out} ({t.min().date()} to {t.max().date()})")
    print(f"late rate {df['late'].mean():.3f}, median delivery {df['delivery_days'].median():.1f} days, "
          f"repeat customers {(df['customer_unique_id'].duplicated()).mean():.3f}")


if __name__ == "__main__":
    main()
