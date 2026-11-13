#!/usr/bin/env bash
# Training-size curve under the temporal split.
# Usage: bash scripts/learning_curve.sh dataco late_delivery
set -euo pipefail
DS=${1:?dataset}; TASK=${2:?task}
for N in 500 2000 10000 50000; do
  tfmbench --dataset "$DS" --task "$TASK" --split temporal \
    --models linear xgboost lightgbm catboost tabpfn tabicl \
    --max-train-rows "$N" --out "results/${DS}_${TASK}_curve.csv"
done
