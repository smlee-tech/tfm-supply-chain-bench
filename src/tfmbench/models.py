"""Model registry.

GBDTs are tuned with Optuna on a validation split (same budget for each), then refit on
the full training window. TFMs run with default settings: no tuning. Optional packages
(tabpfn, tabicl) are imported lazily; a missing package raises ModelUnavailable and the
runner skips that model with a message.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


class ModelUnavailable(RuntimeError):
    pass


def _filter_kwargs(cls, kwargs: dict) -> dict:
    """Keep only constructor arguments the installed version accepts."""
    params = inspect.signature(cls.__init__).parameters
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


def _device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


# ---------- builders: (task_type, seed, params, cat_idx) -> estimator ----------

def _linear(task_type, seed, params, cat_idx):
    from sklearn.linear_model import LogisticRegression, Ridge

    if task_type == "classification":
        return LogisticRegression(max_iter=2000)
    return Ridge(alpha=1.0)


def _xgb(task_type, seed, params, cat_idx):
    try:
        from xgboost import XGBClassifier, XGBRegressor
    except ImportError as e:
        raise ModelUnavailable("pip install xgboost") from e
    base = dict(tree_method="hist", random_state=seed, n_jobs=-1)
    cls = XGBClassifier if task_type == "classification" else XGBRegressor
    return cls(**base, **params)


def _lgbm(task_type, seed, params, cat_idx):
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
    except ImportError as e:
        raise ModelUnavailable("pip install lightgbm") from e
    base = dict(random_state=seed, verbose=-1, subsample_freq=1)
    cls = LGBMClassifier if task_type == "classification" else LGBMRegressor
    return cls(**base, **params)


def _catboost(task_type, seed, params, cat_idx):
    try:
        from catboost import CatBoostClassifier, CatBoostRegressor
    except ImportError as e:
        raise ModelUnavailable("pip install catboost") from e
    base = dict(random_seed=seed, verbose=0, allow_writing_files=False)
    cls = CatBoostClassifier if task_type == "classification" else CatBoostRegressor
    return cls(**base, **params)


def _tabpfn(task_type, seed, params, cat_idx):
    try:
        from tabpfn import TabPFNClassifier, TabPFNRegressor
    except ImportError as e:
        raise ModelUnavailable("pip install tabpfn (see README for weights/license)") from e
    cls = TabPFNClassifier if task_type == "classification" else TabPFNRegressor
    kwargs = dict(device=_device(), random_state=seed,
                  categorical_features_indices=cat_idx or None,
                  ignore_pretraining_limits=True)
    return cls(**_filter_kwargs(cls, kwargs))


def _tabicl(task_type, seed, params, cat_idx):
    try:
        import tabicl
    except ImportError as e:
        raise ModelUnavailable("pip install tabicl") from e
    name = "TabICLClassifier" if task_type == "classification" else "TabICLRegressor"
    cls = getattr(tabicl, name, None)
    if cls is None:
        raise ModelUnavailable(f"installed tabicl has no {name}")
    kwargs = dict(device=_device(), random_state=seed)
    return cls(**_filter_kwargs(cls, kwargs))


# ---------- Optuna search spaces ----------

def _space_xgb(trial):
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 1500, log=True),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 10),
        min_child_weight=trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    )


def _space_lgbm(trial):
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 1500, log=True),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 100, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    )


def _space_catboost(trial):
    return dict(
        iterations=trial.suggest_int("iterations", 200, 2000, log=True),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        depth=trial.suggest_int("depth", 4, 10),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
    )


@dataclass
class ModelSpec:
    name: str
    family: str  # "linear" or "tree" -> chooses the preprocessor
    build: Callable[..., Any]
    space: Callable[[Any], dict] | None = None  # None = no tuning
    is_tfm: bool = False
    defaults: dict = field(default_factory=dict)


REGISTRY: dict[str, ModelSpec] = {
    "linear": ModelSpec("linear", "linear", _linear),
    "xgboost": ModelSpec("xgboost", "tree", _xgb, _space_xgb),
    "lightgbm": ModelSpec("lightgbm", "tree", _lgbm, _space_lgbm),
    "catboost": ModelSpec("catboost", "tree", _catboost, _space_catboost),
    "tabpfn": ModelSpec("tabpfn", "tree", _tabpfn, None, is_tfm=True),
    "tabicl": ModelSpec("tabicl", "tree", _tabicl, None, is_tfm=True),
}


def predict_scores(model, X: np.ndarray, task_type: str) -> np.ndarray:
    """Probability of class 1 for classification, point prediction for regression."""
    if task_type == "classification":
        return np.asarray(model.predict_proba(X))[:, 1]
    return np.asarray(model.predict(X), dtype=float).ravel()


def tune(spec: ModelSpec, task_type: str, seed: int, cat_idx, X_tr, y_tr, X_va, y_va,
         n_trials: int) -> dict:
    """Return the best params found by Optuna on the validation split."""
    if spec.space is None or n_trials <= 0:
        return {}
    import optuna
    from sklearn.metrics import log_loss, mean_squared_error

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        model = spec.build(task_type, seed, spec.space(trial), cat_idx)
        model.fit(X_tr, y_tr)
        s = predict_scores(model, X_va, task_type)
        if task_type == "classification":
            return log_loss(y_va, np.clip(s, 1e-7, 1 - 1e-7), labels=[0, 1])
        return float(np.sqrt(mean_squared_error(y_va, s)))

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params
