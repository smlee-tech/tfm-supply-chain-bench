# tfm-supply-chain-bench

Do tabular foundation models (TFMs) beat tuned gradient-boosted trees on supply chain
prediction when the split respects time? This repository benchmarks TabPFN-3.5 and
TabICLv2 against XGBoost, LightGBM and CatBoost on public supply chain data, under random,
grouped and temporal splits, and reports where each model wins and where it falls short.
TabPFN-2.5 is kept as a reference column, to show how much of the gap the newer
checkpoint closes on the same tasks.

> Status: experiment scaffold. Results will be added as they are produced.

## Research questions

1. **Accuracy:** on regression and classification tasks, do TFMs beat tuned GBDTs?
2. **Split effect:** how much does performance change from a random split to a temporal
   split (train on the past, predict the future)? Most tabular benchmarks, including
   TabArena v0.1, cover IID data only. The change is reported in two steps, because a
   random split of order lines also leaks orders across train and test:
   random → grouped (whole orders kept together) is the group-leakage effect, and
   grouped → temporal is the effect of time.
3. **Failure regions:** when do TFMs lose: training size, high-cardinality categoricals,
   distribution shift over time?
4. **Cost:** what does each model cost in fit and prediction time for the accuracy it gets?

## Data

```bash
bash scripts/download_data.sh            # DataCo and Olist into data/raw/ (not committed)
python scripts/make_synthetic.py         # synthetic control
```

| Dataset (`--dataset`) | Rows | Regression task | Classification task | Role |
| --- | --- | --- | --- | --- |
| Olist Brazilian e-commerce (`olist`) | 96,470 delivered orders, 2016-09 to 2018-08 | `delivery_days` | `late_delivery` | main real dataset |
| DataCo Smart Supply Chain (`dataco`) | 180,519 order lines, 2015-01 to 2018-01 | `shipping_days` | `late_delivery` | caution case, see below |
| Synthetic order lines (`synth`) | set by `--n-orders` | `shipping_days` | `late_delivery` | control with known truth |
| USAID shipment pricing, SCMS (`usaid`) | not downloaded | `freight_cost` | `late_delivery` | on hold: the official host no longer resolves |

- **Olist.** `scripts/prepare_olist.py` joins orders, items, products, sellers, customers,
  payments and zip-code locations into one row per delivered order. The prediction is
  made at purchase: features are the basket, seller, destination, distance, price,
  freight, payment and the promised delivery time. The late rate moves from about 3% in
  2017 to 13% in early 2018 and back to 4%, so the temporal split has real drift to
  measure. Source: the original Kaggle release (`olistbr/brazilian-ecommerce`,
  CC BY-NC-SA 4.0, non-commercial), downloaded from a copy on
  [Hugging Face](https://huggingface.co/datasets/bulutttt/olist-raw-data).
  Confirm the license on the Kaggle page before publishing results.
- **DataCo.** [Mendeley Data, version 5](https://data.mendeley.com/datasets/8gx2fvg2k6/5)
  (DOI 10.17632/8gx2fvg2k6.5, listed as CC BY 4.0). The file behaves like generated
  data: within Standard and Second Class shipping, the real shipping days are spread
  evenly over 2 to 6 (20% each), so lateness is the shipping mode plus noise and
  models are expected to tie. From October 2017 only 13 of the 100 earlier products
  remain. It is kept to show what a benchmark on such data looks like, not to rank models.
- **Synthetic.** A control with a known generating process, not evidence about real
  supply chains: `--drift 0` gives a stationary world where the grouped and temporal
  splits should agree, and the printed oracle AUC is the ceiling any model can approach.
- **Group columns** for the grouped split: customer for Olist (3% of orders are repeat
  customers), `Order Id` for DataCo, order for the synthetic data.
- **Label delay.** A training order placed just before a cutoff may only be delivered
  after it, so its label would not be known yet in practice. Olist deliveries take a
  median of 10 days; no gap is left between train and test yet.

**Leakage control.** Every task lists `leakage_cols`: columns only known after the
prediction time (actual ship date, delivery status, actual freight). They are always
dropped, except in the leakage-injection ablation, which adds them back on purpose to
show how much a leak inflates scores.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"      # baselines + GBDTs + tests
pip install -e ".[tfm]"      # tabpfn 10.0.0 and tabicl 2.2.0, pinned (a GPU is strongly recommended)
pytest -q
```

| `--models` name | Model | Checkpoint | Weights license |
| --- | --- | --- | --- |
| `tabpfn` | TabPFN-3.5 (main) | `tabpfn-v3.5-20260909`, pinned in `models.py` | [TabPFN-3.5 license](https://huggingface.co/Prior-Labs/tabpfn_3_5/blob/main/LICENSE), non-commercial |
| `tabpfn-2.5` | TabPFN-2.5 (reference) | pinned in `models.py` | its own non-commercial license, see the [Prior Labs models page](https://docs.priorlabs.ai/models#tabpfn-model-license) |
| `tabicl` | TabICLv2 | `tabicl-classifier-v2-20260212`, the default of tabicl 2.2.0 | see the [TabICL model page](https://huggingface.co/jingang/TabICL) |

The TabPFN weights are licensed separately from the `tabpfn` package (Apache 2.0), and
each checkpoint has its own license document, so 3.5 and 2.5 are accepted separately. The
first run opens a browser to accept the license; on a headless machine set `TABPFN_TOKEN`.

TabPFN-3.5 accepts up to 1,000,000 training rows, and TabPFN-2.5 is recommended up to
50,000. The commands below still cap training at 50,000 rows so that both checkpoints see
the same data; how far TabPFN-3.5 goes without the cap on the available GPU has not been
measured yet.

## Running

```bash
# Main comparison, temporal split, 3 rolling-origin cutoffs x 5 seeds
tfmbench --dataset olist --task late_delivery --split temporal --n-folds 3 \
  --models linear xgboost lightgbm catboost tabpfn tabpfn-2.5 tabicl --max-train-rows 50000

# Same, grouped and random splits (to measure the split effect in two steps)
tfmbench --dataset olist --task late_delivery --split grouped \
  --models linear xgboost lightgbm catboost tabpfn tabpfn-2.5 tabicl --max-train-rows 50000
tfmbench --dataset olist --task late_delivery --split random \
  --models linear xgboost lightgbm catboost tabpfn tabpfn-2.5 tabicl --max-train-rows 50000

# GBDTs at default settings (the no-tuning comparison with the TFMs)
tfmbench --dataset olist --task late_delivery --split temporal --n-trials 0 \
  --models xgboost lightgbm catboost --max-train-rows 50000

# Ablations
bash scripts/learning_curve.sh olist late_delivery           # 500 / 2k / 10k / 50k rows
tfmbench --dataset olist --task late_delivery --keep-ids     # high-cardinality IDs kept
tfmbench --dataset olist --task late_delivery --inject-leak  # deliberate leakage

python scripts/plot_results.py results/olist_late_delivery_curve.csv
```

## Protocol

| Item | Setting |
| --- | --- |
| Splits | random (rows shuffled), grouped (whole orders or shipments shuffled, none on both sides), and temporal (every training row strictly before every test row); test = 20% |
| Temporal cutoffs | rolling origin: the last 60% of the timeline is cut into 3 consecutive 20% test blocks; each is predicted from everything before it (training window 40%, 60%, 80%) |
| Validation | 20% of the training window, same split type as the test split |
| GBDT tuning | Optuna TPE, 50 trials per model on validation (log loss / RMSE), then refit on the full training window; also run untuned (`--n-trials 0`) |
| TFMs | TabPFN-3.5 (main), TabPFN-2.5 (reference) and TabICLv2, each with a pinned checkpoint; default settings, no tuning |
| Preprocessing | fitted on training data only; categoricals ordinal-encoded for trees and TFMs, one-hot for linear baselines |
| Categoricals | XGBoost, LightGBM, CatBoost and TabPFN (3.5 and 2.5) are told which columns are categorical and use their native handling; TabICLv2 receives the codes as they are |
| Classification metrics | ROC-AUC, log loss, Brier score, expected calibration error (ECE) |
| Regression metrics | RMSE, MAE, R² |
| Cost | tuning (`tune_s`), final fit (`fit_s`) and predict (`predict_s`) wall time, hardware recorded |
| Uncertainty | 5 seeds; 95% bootstrap CI of the primary metric on test rows. For the temporal split the spread over cutoffs is reported separately from the spread over seeds: all seeds share the same test rows, so seeds alone would look too stable |

Results are reported as they come out, including tasks where TFMs lose.

## Branches

- `main`: data loading, splits, models, metrics, tests
- `exp/temporal-split`: main random / grouped / temporal comparison
- `exp/ablations`: learning curve, categorical, leakage-injection experiments

Each experiment branch is merged into `main` by pull request once its results are final.

## Results

_To be added._

## Notes

Personal research project using public data and public models only. Not affiliated with
or endorsed by any employer or model provider.

## License

Code: MIT (see `LICENSE`). Datasets and model weights keep their own licenses.
