import numpy as np
import pandas as pd

from tfmbench.splits import random_split, subsample, temporal_split, validation_split


def test_temporal_split_is_strictly_ordered():
    t = pd.Series(pd.date_range("2020-01-01", periods=1000, freq="D")).sample(frac=1, random_state=0)
    tr, te = temporal_split(t.reset_index(drop=True), 0.2)
    tt = t.reset_index(drop=True)
    assert tt.iloc[tr].max() < tt.iloc[te].min()
    assert abs(len(te) / 1000 - 0.2) < 0.01


def test_random_split_partitions():
    tr, te = random_split(100, 0.25, seed=1)
    assert len(set(tr) & set(te)) == 0 and len(tr) + len(te) == 100


def test_validation_split_inside_train():
    t = pd.Series(pd.date_range("2021-01-01", periods=500, freq="h"))
    tr, _ = temporal_split(t, 0.2)
    fit, val = validation_split(tr, t, "temporal", 0.2, seed=0)
    assert set(fit) | set(val) == set(tr)
    assert t.iloc[fit].max() < t.iloc[val].min()


def test_subsample():
    idx = np.arange(1000)
    s = subsample(idx, 100, seed=0)
    assert len(s) == 100 and set(s) <= set(idx)
