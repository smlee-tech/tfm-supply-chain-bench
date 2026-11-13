"""Preprocessing, fitted on training data only.

- "tree" (GBDTs, TFMs): categorical columns -> ordinal codes (unknown = -1), numeric kept
  as is; missing values are left for the model to handle.
- "linear" (baselines): numeric -> median impute + scale; categorical -> capped one-hot.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


def column_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    cat = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    num = [c for c in X.columns if c not in cat]
    return num, cat


def make_preprocessor(kind: str, X: pd.DataFrame) -> ColumnTransformer:
    num, cat = column_types(X)
    if kind == "tree":
        cat_pipe = Pipeline([
            ("to_str", _ToStr()),
            ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                   encoded_missing_value=-1)),
        ])
        return ColumnTransformer(
            [("num", "passthrough", num), ("cat", cat_pipe, cat)],
            verbose_feature_names_out=False,
        )
    if kind == "linear":
        num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
        cat_pipe = Pipeline([
            ("to_str", _ToStr()),
            ("oh", OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=30,
                                 sparse_output=False)),
        ])
        return ColumnTransformer(
            [("num", num_pipe, num), ("cat", cat_pipe, cat)],
            verbose_feature_names_out=False,
        )
    raise ValueError(f"Unknown preprocessor kind: {kind}")


class _ToStr:
    """Cast categorical columns to strings so mixed types and NaN encode consistently."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame(X).astype("string").fillna("__missing__").to_numpy(dtype=object)

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self


def categorical_indices(X: pd.DataFrame) -> list[int]:
    """Column positions of categorical features AFTER the 'tree' preprocessor (num first, then cat)."""
    num, cat = column_types(X)
    return list(range(len(num), len(num) + len(cat)))


def to_float(a) -> np.ndarray:
    return np.asarray(a, dtype=float)
