# tfm-supply-chain-bench

Do tabular foundation models (TFMs) beat tuned gradient-boosted trees on supply chain
prediction when the split respects time? This repository benchmarks TabPFN-3.5 and
TabICLv2 against XGBoost, LightGBM and CatBoost on public supply chain data, under random,
grouped and temporal splits, and reports where each model wins and where it falls short.
TabPFN-2.5 is kept as a reference column, to show how much of the gap the newer
checkpoint closes on the same tasks.

> Status: the main model comparison has not been run under the full protocol yet. The first
> result is an exploratory study of label delay on the `exp/label-delay` branch, see
> [Results](#results).

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
bash scripts/download_data.sh            # DataCo, Olist and Criteo into data/raw/ (not committed)
python scripts/make_synthetic.py         # synthetic control
```

| Dataset (`--dataset`) | Rows | Regression task | Classification task | Role |
| --- | --- | --- | --- | --- |
| Olist Brazilian e-commerce (`olist`) | 96,470 delivered orders, 2016-09 to 2018-08 | `delivery_days` | `late_delivery` | main real dataset |
| DataCo Smart Supply Chain (`dataco`) | 180,519 order lines, 2015-01 to 2018-01 | `shipping_days` | `late_delivery` | caution case, see below |
| NYC 311 housing requests (`nyc311_hpd`) | 98,305 requests to HPD, 2020 to 2021 (a 10% sample) | none | `slow_14d` | second dataset for the label-delay study, not supply chain |
| Criteo attribution (`criteo_attribution`) | 118,875 ad clicks over 31 days (a 2% sample of users) | none | `converted_7d` | third dataset for the label-delay study, not supply chain |
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
- **NYC 311.** `scripts/prepare_nyc311.py` downloads a sample of the service requests of one
  agency from [NYC Open Data](https://data.cityofnewyork.us/d/erm2-nwe9) (requests whose
  key ends in 7). The task `slow_14d` asks, when a request is filed, whether it will still
  be open 14 days later. The 14-day threshold is set by this repository (the official due
  date is filled for 0.4% of rows), so it is a constructed task. It is here because a fast
  request is labelled when it closes and a slow one only after 14 days, the same label
  mechanism as on Olist.
- **Criteo attribution.** `scripts/prepare_criteo.py` keeps the clicked impressions of the
  [Criteo Attribution Modeling for Bidding Dataset](https://huggingface.co/datasets/criteo/criteo-attribution-dataset)
  (CC BY-NC-SA 4.0, non-commercial). The task `converted_7d` asks, at the click, whether a
  conversion follows within 7 days. Here the positive label arrives first (at the
  conversion) and the negative one after 7 days, the opposite direction to Olist. The
  timestamps are relative, so the dates sit on an arbitrary origin, and the 7-day window is
  set by this repository.
- **Synthetic.** A control with a known generating process, not evidence about real
  supply chains: `--drift 0` gives a stationary world where the grouped and temporal
  splits should agree, and the printed oracle AUC is the ceiling any model can approach.
- **Group columns** for the grouped split: customer for Olist (3% of orders are repeat
  customers), `Order Id` for DataCo, order for the synthetic data.
- **Label delay.** A training order placed just before a cutoff may only be delivered
  after it, so its label would not be known yet in practice. On Olist an on-time label
  arrives at delivery and a late label when the promised date passes, so the recent orders
  with a known label are mostly the on-time ones. The default temporal split ignores this
  (`--label-policy oracle`). `--label-policy known` trains only on labels that have arrived
  by the cutoff, and `--label-policy gap` also leaves out the last `--label-gap-days`
  (default 30) before the cutoff. The label time of a task is set by `label_time_cols` in
  `configs/datasets.yaml`.

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
tfmbench --dataset olist --task late_delivery --label-policy known  # only labels known at the cutoff
tfmbench --dataset olist --task late_delivery --max-train-rows 10000 --train-sample recent  # latest rows instead of a random sample
tfmbench --dataset olist --task late_delivery --drop-cols pred_month pred_dayofyear  # without seasonal calendar features

python scripts/plot_results.py results/olist_late_delivery_curve.csv
```

Results are appended to `results/<dataset>_<task>.csv` unless `--out` is given.

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

`main` holds data loading, splits, models, metrics and tests. `exp/label-delay` holds the
label-delay study described under Results. Other experiments will run on their own
branches (`exp/temporal-split` for the random / grouped / temporal comparison,
`exp/ablations` for the learning curve, categorical and leakage-injection runs), each
merged into `main` by pull request once its results are final.

## Results

The main comparison (all models, tuned GBDTs, 5 seeds, random / grouped / temporal) is not
done. The one result so far is exploratory.

### Label delay: recent rows with a known label are a biased context (exploratory)

**Question.** A tabular foundation model adapts to drift by refreshing its context with
recent rows, and a GBDT by retraining on them. But a row can only be used once its label
has arrived, and on these tasks the arrival time depends on the outcome: an on-time order
is labelled at delivery, a late one only when the promised date has passed. What does the
usual temporal split, which treats every label before the cutoff as known, hide?

**Setup.** Rolling-origin temporal split with 10 cutoffs (each test block is 6% of the
rows), 10,000 training rows, 3 seeds, no tuning (LightGBM at default settings). The grid is

- label policy: `oracle` (every label before the cutoff, the usual split), `known` (only
  labels that had arrived by the cutoff), `gap` (as `known`, and without the last 30 days
  before the cutoff on Olist, 14 days on NYC 311);
- training sample: the 10,000 most `recent` usable rows, or a `random` 10,000 of all usable rows;
- features: all, or without the two calendar features that encode the position in the
  year (`pred_month`, `pred_dayofyear`).

Mean ROC-AUC over the 10 cutoffs, most recent 10,000 rows (worst cutoff in brackets):

| Dataset | Model | Features | oracle | known | gap |
| --- | --- | --- | --- | --- | --- |
| Olist | TabPFN-3.5 | all | 0.734 (0.660) | 0.619 (0.445) | 0.702 (0.603) |
| Olist | TabPFN-3.5 | no month / day-of-year | 0.733 (0.625) | 0.715 (0.654) | 0.699 (0.575) |
| Olist | LightGBM | all | 0.688 (0.607) | 0.617 (0.443) | 0.657 (0.566) |
| Olist | LightGBM | no month / day-of-year | 0.693 (0.559) | 0.677 (0.608) | 0.656 (0.535) |
| Olist | linear | all | 0.703 (0.618) | 0.692 (0.583) | 0.699 (0.627) |
| NYC 311 | TabPFN-3.5 | all | 0.802 (0.639) | 0.720 (0.513) | 0.800 (0.641) |
| NYC 311 | TabPFN-3.5 | no month / day-of-year | 0.800 (0.639) | 0.799 (0.640) | 0.798 (0.640) |
| NYC 311 | LightGBM | all | 0.789 (0.628) | 0.699 (0.516) | 0.784 (0.626) |
| NYC 311 | LightGBM | no month / day-of-year | 0.791 (0.626) | 0.790 (0.625) | 0.791 (0.631) |
| NYC 311 | linear | all | 0.796 (0.639) | 0.776 (0.638) | 0.794 (0.639) |

![ROC-AUC of TabPFN-3.5 per cutoff, by label policy and feature set](docs/label_delay.png)

**What the numbers say.**

1. *The usual temporal split is optimistic, and the naive fix is worse than no fix.* With a
   context of the most recent rows whose label has arrived, TabPFN-3.5 loses 0.115 ROC-AUC
   on Olist (95% interval over cutoffs 0.055 to 0.179) and 0.082 on NYC 311 (0.040 to
   0.139), and falls to chance level at the worst cutoff (0.445 and 0.513). Default
   LightGBM loses 0.071 and 0.090.
2. *The damage goes through the calendar features.* Without month and day-of-year the loss
   shrinks to under 0.02 on Olist and to 0.001 on NYC 311, for both model families. The rows from the
   last weeks before a cutoff that already have a label are almost all on-time ones, so a model
   that sees the date learns "recent means on time" and applies it to the test period.
3. *It is not specific to foundation models.* TabPFN-3.5 and LightGBM break in the same
   place by a similar amount; the linear model moves by 0.02 or less.
4. *Waiting for labels is free on one dataset and costs accuracy on the other.* Leaving out
   the last days before the cutoff recovers everything on NYC 311 (0.800 against 0.802). On
   Olist, where the late rate moves quickly, it still costs 0.033 (interval -0.008 to 0.078,
   so not separated from zero with 10 cutoffs).
5. *Model ranking.* TabPFN-3.5 is ahead of default LightGBM by 0.04 to 0.05 on Olist and
   0.01 to 0.02 on NYC 311, except in the collapse in point 1, where the lead on Olist
   disappears (0.002). Against the linear model the difference is within about 0.03
   either way, and in the collapse TabPFN-3.5 is 0.073 (Olist) and 0.056 (NYC 311) behind it.

**Limits.** Exploratory: LightGBM is untuned, the training set is fixed at 10,000 rows,
there are two datasets, and the NYC 311 task is constructed here. The 10 cutoffs are
consecutive and share training data, so the intervals understate the uncertainty. Which
usable policy is best on Olist (`known` without calendar features, 0.715, or `gap`, 0.702)
is not settled by these runs. Tuned runs with 5 seeds and a third dataset are in progress.

**Relation to earlier work.** That delayed feedback biases recent training data is well
known in conversion prediction (Chapelle 2014; Yang et al. 2021, arXiv 2012.03245; Gu et
al. 2021, arXiv 2104.14121) and fraud detection (Dal Pozzolo et al. 2015). Work on tabular
foundation models under label delay so far uses a fixed delay that is the same for every
row (arXiv 2610.05352), which cannot produce this bias. What this study adds is a
measurement on business tables with real label arrival times, for an in-context model and
a GBDT side by side, and the observation that the calendar features carry the damage.

**Reproduce.** The tables and the figure come from the CSVs in `results/label_delay/`:

```bash
python scripts/prepare_nyc311.py           # NYC 311 sample; Olist as in "Data"
tfmbench --dataset olist --task late_delivery --split temporal --n-folds 10 --test-size 0.06 \
  --seeds 0 1 2 --n-trials 0 --models tabpfn --max-train-rows 10000 --train-sample recent \
  --label-policy known --label-gap-days 30 --out results/label_delay/untuned_olist_late_delivery.csv
python scripts/summarize_label_delay.py results/label_delay/untuned_*.csv --plot docs/label_delay.png
```

On macOS run TabPFN in its own process: after LightGBM has run in the same process the two
OpenMP runtimes can deadlock.

## Notes

Personal research project using public data and public models only. Not affiliated with
or endorsed by any employer or model provider.

## License

Code: MIT (see `LICENSE`). Datasets and model weights keep their own licenses.
