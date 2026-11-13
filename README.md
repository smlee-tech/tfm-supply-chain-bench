# tfm-supply-chain-bench

Do tabular foundation models (TFMs) beat tuned gradient-boosted trees on supply chain
prediction when the split respects time? This repository benchmarks TabPFN-2.5 and
TabICL against XGBoost, LightGBM and CatBoost on public supply chain data, under both
random and temporal splits, and reports where each model wins and where it falls short.

> Status: experiment scaffold. Results will be added as they are produced.

## Research questions

1. **Accuracy:** on regression and classification tasks, do TFMs beat tuned GBDTs?
2. **Split effect:** how much does performance change from a random split to a temporal
   split (train on the past, predict the future)? Most tabular benchmarks, including
   TabArena v0.1, cover IID data only.
3. **Failure regions:** when do TFMs lose: training size, high-cardinality categoricals,
   distribution shift over time?
4. **Cost:** what does each model cost in fit and prediction time for the accuracy it gets?

## Data

Download the files yourself and place them in `data/raw/` (not committed).

| Dataset | File expected in `data/raw/` | Regression task | Classification task |
| --- | --- | --- | --- |
| DataCo Smart Supply Chain | `DataCoSupplyChainDataset.csv` | `shipping_days` | `late_delivery` |
| USAID Supply Chain Shipment Pricing (SCMS) | `SCMS_Delivery_History_Dataset.csv` | `freight_cost` | `late_delivery` |

- Check and record each dataset's license here before publishing results.
- Then confirm the column names used in `configs/datasets.yaml`:
  `python scripts/inspect_columns.py dataco`

**Leakage control.** Every task lists `leakage_cols`: columns only known after the
prediction time (actual ship date, delivery status, actual freight). They are always
dropped, except in the leakage-injection ablation, which adds them back on purpose to
show how much a leak inflates scores.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"      # baselines + GBDTs + tests
pip install -e ".[tfm]"      # TabPFN and TabICL (a GPU is strongly recommended)
pytest -q
```

TabPFN-2.5 open weights are for research and non-commercial use; see the
[Prior Labs model overview](https://docs.priorlabs.ai/models/overview). Its recommended
range is up to 50,000 training rows, so larger datasets are studied with a learning curve.

## Running

```bash
# Main comparison, temporal split, 5 seeds
tfmbench --dataset dataco --task late_delivery --split temporal \
  --models linear xgboost lightgbm catboost tabpfn tabicl --max-train-rows 50000

# Same, random split (to measure the split effect)
tfmbench --dataset dataco --task late_delivery --split random \
  --models linear xgboost lightgbm catboost tabpfn tabicl --max-train-rows 50000

# Ablations
bash scripts/learning_curve.sh dataco late_delivery          # 500 / 2k / 10k / 50k rows
tfmbench --dataset dataco --task late_delivery --keep-ids     # high-cardinality IDs kept
tfmbench --dataset dataco --task late_delivery --inject-leak  # deliberate leakage

python scripts/plot_results.py results/dataco_late_delivery_curve.csv
```

## Protocol

| Item | Setting |
| --- | --- |
| Splits | random, and temporal (every training row strictly before every test row); test = 20% |
| Validation | 20% of the training window, same split type as the test split |
| GBDT tuning | Optuna TPE, 50 trials per model on validation (log loss / RMSE), then refit on the full training window |
| TFMs | default settings, no tuning |
| Preprocessing | fitted on training data only; categoricals ordinal-encoded for trees and TFMs, one-hot for linear baselines |
| Classification metrics | ROC-AUC, log loss, Brier score, expected calibration error (ECE) |
| Regression metrics | RMSE, MAE, R² |
| Cost | fit and predict wall time, hardware recorded |
| Uncertainty | 5 seeds; 95% bootstrap CI of the primary metric on test rows |

Results are reported as they come out, including tasks where TFMs lose.

## Branches

- `main`: data loading, splits, models, metrics, tests
- `exp/temporal-split`: main random-vs-temporal comparison
- `exp/ablations`: learning curve, categorical, leakage-injection experiments

Each experiment branch is merged into `main` by pull request once its results are final.

## Results

_To be added._

## Notes

Personal research project using public data and public models only. Not affiliated with
or endorsed by any employer or model provider.

## License

Code: MIT (see `LICENSE`). Datasets and model weights keep their own licenses.
