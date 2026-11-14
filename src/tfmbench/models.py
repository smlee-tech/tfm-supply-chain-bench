"""Model registry.

GBDTs are tuned with Optuna on a validation split (same budget for each), then refit on
the full training window. TFMs run with default settings: no tuning. Every GBDT is told
which columns are categorical and uses its native categorical handling, as TabPFN does;
otherwise the ordinal codes would be read as ordered numbers. Optional packages
(tabpfn, tabicl) are imported lazily; a missing package raises ModelUnavailable and the
runner skips that model with a message.
"""
from __future__ import annotations

import inspect
import os
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

        if torch.cuda.is_available():
            return "cuda"
        if os.environ.get("TFMBENCH_DEVICE"):  # e.g. "mps" on Apple silicon, or "cpu"
            return os.environ["TFMBENCH_DEVICE"]
        return "cpu"
    except ImportError:
        return "cpu"


class _NativeCategorical:
    """Fit/predict wrapper that hands the categorical columns to a GBDT natively.

    Input is the float matrix from the "tree" preprocessor: categorical columns hold
    ordinal codes >= 0, and -1 for a category not seen when the encoder was fitted.
    """

    def __init__(self, model, cat_idx: list[int], lib: str):
        self.model, self.cat_idx, self.lib = model, list(cat_idx), lib

    def _prep(self, X):
        X = np.asarray(X, dtype=float)
        if self.lib == "xgboost":  # negative codes are invalid; NaN means "missing"
            X = X.copy()
            block = X[:, self.cat_idx]
            block[block < 0] = np.nan
            X[:, self.cat_idx] = block
            return X
        if self.lib == "catboost":  # needs integer (not float) categorical columns
            import pandas as pd

            df = pd.DataFrame(X)
            df[self.cat_idx] = df[self.cat_idx].astype(int)
            return df
        return X  # lightgbm: negative codes are treated as missing

    def fit(self, X, y):
        Xp = self._prep(X)
        if self.lib == "xgboost":
            types = ["c" if i in set(self.cat_idx) else "q" for i in range(Xp.shape[1])]
            self.model.set_params(enable_categorical=True, feature_types=types)
            self.model.fit(Xp, y)
        elif self.lib == "lightgbm":
            self.model.fit(Xp, y, categorical_feature=self.cat_idx)
        else:
            self.model.fit(Xp, y, cat_features=self.cat_idx)
        return self

    def predict(self, X):
        return self.model.predict(self._prep(X))

    def predict_proba(self, X):
        return self.model.predict_proba(self._prep(X))


def _with_categoricals(model, cat_idx, lib: str):
    return _NativeCategorical(model, cat_idx, lib) if cat_idx else model


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
    return _with_categoricals(cls(**base, **params), cat_idx, "xgboost")


def _lgbm(task_type, seed, params, cat_idx):
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
    except ImportError as e:
        raise ModelUnavailable("pip install lightgbm") from e
    base = dict(random_state=seed, verbose=-1, subsample_freq=1)
    cls = LGBMClassifier if task_type == "classification" else LGBMRegressor
    return _with_categoricals(cls(**base, **params), cat_idx, "lightgbm")


def _catboost(task_type, seed, params, cat_idx):
    try:
        from catboost import CatBoostClassifier, CatBoostRegressor
    except ImportError as e:
        raise ModelUnavailable("pip install catboost") from e
    base = dict(random_seed=seed, verbose=0, allow_writing_files=False)
    cls = CatBoostClassifier if task_type == "classification" else CatBoostRegressor
    return _with_categoricals(cls(**base, **params), cat_idx, "catboost")


def _tabpfn_builder(version: str):
    """Builder for one pinned TabPFN checkpoint, e.g. "v3.5" or "v2.5".

    The version is always pinned, never left to the package default, so the name in the
    results stays true when a newer tabpfn release changes its default checkpoint.
    """

    def build(task_type, seed, params, cat_idx):
        try:
            from tabpfn import TabPFNClassifier, TabPFNRegressor
            from tabpfn.constants import ModelVersion
        except ImportError as e:
            raise ModelUnavailable(
                "pip install tabpfn (see README for weights/license)") from e
        try:
            model_version = ModelVersion(version)
        except ValueError as e:
            raise ModelUnavailable(
                f"installed tabpfn has no checkpoint {version}; "
                'pip install -e ".[tfm]" for the pinned version') from e
        cls = TabPFNClassifier if task_type == "classification" else TabPFNRegressor
        kwargs = dict(device=_device(), random_state=seed,
                      categorical_features_indices=cat_idx or None,
                      ignore_pretraining_limits=True)
        return cls.create_default_for_version(model_version, **_filter_kwargs(cls, kwargs))

    return build


def _tabicl(task_type, seed, params, cat_idx):
    try:
        import tabicl
    except ImportError as e:
        raise ModelUnavailable("pip install tabicl") from e
    name = "TabICLClassifier" if task_type == "classification" else "TabICLRegressor"
    cls = getattr(tabicl, name, None)
    if cls is None:
        raise ModelUnavailable(f"installed tabicl has no {name}")
    # checkpoint_version is left at the package default: TabICLv2 in the pinned tabicl 2.2.0
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
    # "tabpfn" is the main TFM (TabPFN-3.5); "tabpfn-2.5" is a reference column that
    # shows how much the newer checkpoint changed the result on the same task.
    "tabpfn": ModelSpec("tabpfn", "tree", _tabpfn_builder("v3.5"), None, is_tfm=True),
    "tabpfn-2.5": ModelSpec("tabpfn-2.5", "tree", _tabpfn_builder("v2.5"), None,
                            is_tfm=True),
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
