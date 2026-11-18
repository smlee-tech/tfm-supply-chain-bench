#!/usr/bin/env bash
# Label-delay grid for one dataset: train sample x label policy x calendar features,
# rolling-origin temporal split with 10 cutoffs and 10,000 training rows.
#
# Usage: bash scripts/label_delay_grid.sh <dataset> <task> <gap_days> <n_trials> "<seeds>" <model> [model ...]
#   bash scripts/label_delay_grid.sh olist late_delivery 30 0 "0 1 2" linear lightgbm    # untuned
#   FEATS=nodate adds the grid without any calendar feature (default: all nocal).
#   SAMPLES=recent bash scripts/label_delay_grid.sh olist late_delivery 30 20 "0 1 2 3 4" lightgbm
#
# One process per configuration and seed: a finished configuration is saved and skipped on
# a rerun, and TabPFN never shares a process with LightGBM (on macOS their OpenMP runtimes
# can deadlock). Run TabPFN in a separate call of this script.
# Results go to results/label_delay/{untuned,tuned}_<dataset>_<task>.csv.
set -uo pipefail
ds=$1 task=$2 gap=$3 trials=$4 seeds=$5; shift 5
kind=untuned; [[ $trials -gt 0 ]] && kind=tuned
out=results/label_delay/${kind}_${ds}_${task}.csv
done_dir=results/label_delay/.done; mkdir -p "$done_dir"
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1 TABPFN_NO_BROWSER=1

for seed in $seeds; do
  for feats in ${FEATS:-all nocal}; do
    for sample in ${SAMPLES:-recent random}; do
      for pol in oracle known gap; do
        key="$kind.$ds.$1.$seed.$feats.$sample.$pol"
        [[ -e $done_dir/$key ]] && continue
        drop=()
        [[ $feats == nocal ]] && drop=(--drop-cols pred_month pred_dayofyear)
        [[ $feats == nodate ]] && drop=(--drop-cols pred_month pred_dayofyear pred_dayofweek)
        echo "$(date '+%m-%d %H:%M:%S') start $key"
        if tfmbench --dataset "$ds" --task "$task" --split temporal --n-folds 10 --test-size 0.06 \
            --seeds "$seed" --n-trials "$trials" --models "$@" --max-train-rows 10000 \
            --train-sample "$sample" --label-policy "$pol" --label-gap-days "$gap" \
            ${drop[@]+"${drop[@]}"} --out "$out" 2>&1 | grep --line-buffered -E "^seed=|rror|appended"; then
          touch "$done_dir/$key"
        fi
      done
    done
  done
done
echo "$(date '+%m-%d %H:%M:%S') done $kind $ds $*"
