"""End-to-end run on synthetic order data: leakage handling, splits, models, metrics."""
import numpy as np
import pandas as pd

from tfmbench.data import build_task
from tfmbench.run import run_experiment

CFG = {
    "time_col": "order_date",
    "group_col": "order_id",
    "drop_cols": ["email"],
    "id_cols": ["customer_id", "order_id"],
    "other_date_cols": ["ship_date"],
    "tasks": {
        "late": {"type": "classification", "target": {"kind": "column", "col": "late"},
                 "leakage_cols": ["ship_days", "ship_date"]},
        "days": {"type": "regression", "target": {"kind": "column", "col": "ship_days"},
                 "leakage_cols": ["late", "ship_date"]},
    },
}


def _orders(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    order_date = pd.Timestamp("2022-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D")
    mode = rng.choice(["Standard", "First", "Same Day"], n)
    scheduled = np.select([mode == "Same Day", mode == "First"], [0, 1], 4)
    ship_days = scheduled + rng.poisson(1.0, n)
    return pd.DataFrame({
        "order_date": order_date.astype(str),
        "ship_date": (order_date + pd.to_timedelta(ship_days, unit="D")).astype(str),
        "mode": mode, "scheduled_days": scheduled,
        "qty": rng.integers(1, 10, n), "region": rng.choice(list("ABCDE"), n),
        "customer_id": rng.integers(0, 10_000, n), "email": "x@y.z",
        "order_id": rng.integers(0, n // 3, n),
        "ship_days": ship_days, "late": (ship_days > scheduled).astype(int),
    })


def test_leakage_columns_removed_by_default():
    task = build_task(_orders(), CFG, "late")
    for col in ["ship_days", "ship_date", "late", "email", "customer_id", "order_id",
                "order_date"]:
        assert col not in task.X.columns
    leaky = build_task(_orders(), CFG, "late", inject_leak=True)
    assert "ship_days" in leaky.X.columns and "ship_date" in leaky.X.columns


def test_classification_and_regression_run():
    df = _orders()
    for name in ["late", "days"]:
        task = build_task(df, CFG, name)
        res = run_experiment(task, "temporal", ["linear", "xgboost", "lightgbm", "catboost"],
                             seeds=[0], n_trials=2, n_boot=20, log=lambda *_: None)
        assert set(res["model"]) == {"linear", "xgboost", "lightgbm", "catboost"}
        metric = "roc_auc" if name == "late" else "rmse"
        assert res[metric].notna().all()


def test_leak_inflates_score():
    df = _orders()
    clean = run_experiment(build_task(df, CFG, "late"), "temporal", ["lightgbm"], [0],
                           n_trials=0, n_boot=10, log=lambda *_: None)
    leaky = run_experiment(build_task(df, CFG, "late", inject_leak=True), "temporal",
                           ["lightgbm"], [0], n_trials=0, n_boot=10, log=lambda *_: None)
    assert leaky["roc_auc"].iloc[0] > clean["roc_auc"].iloc[0]


def test_grouped_split_and_rolling_origin_run():
    task = build_task(_orders(), CFG, "late")
    assert task.groups is not None and len(task.groups) == len(task.y)
    grouped = run_experiment(task, "grouped", ["lightgbm"], [0], n_trials=2, n_boot=10,
                             log=lambda *_: None)
    assert grouped["roc_auc"].notna().all() and (grouped["tune_s"] > 0).all()
    rolling = run_experiment(task, "temporal", ["linear", "lightgbm"], [0], n_trials=0,
                             n_boot=10, n_folds=3, log=lambda *_: None)
    assert sorted(rolling["fold"].unique()) == [0, 1, 2]
    assert rolling.groupby("fold")["cutoff"].first().is_monotonic_increasing
    assert (rolling["tune_s"] == 0).all()


def test_gbdts_use_native_categoricals():
    """Unordered categories: a GBDT that reads the codes as numbers needs many more splits."""
    from tfmbench.models import REGISTRY, _NativeCategorical, predict_scores

    rng = np.random.default_rng(0)
    codes = rng.integers(0, 40, 4000).astype(float)
    y = (rng.permutation(40)[codes.astype(int)] % 2).astype(int)  # label depends on category only
    X = np.column_stack([rng.normal(size=4000), codes])
    X_test = X[3000:].copy()
    X_test[:10, 1] = -1  # categories unseen in training must not crash
    for name in ["xgboost", "lightgbm", "catboost"]:
        model = REGISTRY[name].build("classification", 0, {}, [1])
        assert isinstance(model, _NativeCategorical)
        model.fit(X[:3000], y[:3000])
        p = predict_scores(model, X_test, "classification")
        assert ((p[10:] > 0.5) == y[3010:]).mean() > 0.99
        assert isinstance(REGISTRY[name].build("classification", 0, {}, []), type(model.model))
