#!/usr/bin/env bash
# Download the public datasets into data/raw/ (not committed).
# Usage: bash scripts/download_data.sh [dataco] [olist] [criteo]     (default: all)
# Check each dataset's license (README, "Data") before publishing results.
set -euo pipefail
RAW=data/raw
want() { [[ $# -eq 1 ]] || [[ " ${*:2} " == *" $1 "* ]]; }

if want dataco "$@"; then
  # Mendeley Data, DOI 10.17632/8gx2fvg2k6.5
  curl -fL --retry 3 -o "$RAW/DataCoSupplyChainDataset.csv" \
    "https://data.mendeley.com/public-files/datasets/8gx2fvg2k6/files/72784be5-36d3-44fe-b75d-0edbf1999f65/file_downloaded"
  echo "fa6d022ed437155e1a2f0378710602848703c8a7f203f7ff5d77805bf8480aa6  $RAW/DataCoSupplyChainDataset.csv" \
    | shasum -a 256 -c -
fi

if want olist "$@"; then
  # Copy of the original Kaggle release (olistbr/brazilian-ecommerce) on Hugging Face.
  mkdir -p "$RAW/olist"
  for f in olist_orders_dataset olist_order_items_dataset olist_customers_dataset \
           olist_products_dataset olist_sellers_dataset olist_order_payments_dataset \
           olist_geolocation_dataset product_category_name_translation; do
    curl -fL --retry 3 -o "$RAW/olist/$f.csv" \
      "https://huggingface.co/datasets/bulutttt/olist-raw-data/resolve/main/$f.csv"
  done
  python scripts/prepare_olist.py --src "$RAW/olist" --out "$RAW/olist_orders.csv"
fi

if want criteo "$@"; then
  # Criteo Attribution Modeling for Bidding Dataset, CC BY-NC-SA 4.0 (653 MB).
  mkdir -p "$RAW/criteo_attribution"
  curl -fL --retry 8 --retry-all-errors -C - -o "$RAW/criteo_attribution/criteo_attribution_dataset.tsv.gz" \
    "https://huggingface.co/datasets/criteo/criteo-attribution-dataset/resolve/main/criteo_attribution_dataset.tsv.gz"
  python scripts/prepare_criteo.py
fi
