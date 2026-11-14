import numpy as np
import pandas as pd

from tfmbench.splits import (grouped_split, make_folds, random_split, subsample,
                             temporal_folds, temporal_split, validation_split)


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


def test_grouped_split_keeps_groups_together():
    rng = np.random.default_rng(0)
    groups = pd.Series(rng.integers(0, 300, 2000)).astype("object")
    groups.iloc[:5] = None  # missing group = its own group
    tr, te = grouped_split(groups, 0.2, seed=0)
    assert len(set(tr) & set(te)) == 0 and len(tr) + len(te) == 2000
    assert not set(groups.iloc[tr].dropna()) & set(groups.iloc[te].dropna())
    assert 0.1 < len(te) / 2000 < 0.3


def test_grouped_validation_split_keeps_groups_together():
    groups = pd.Series(np.repeat(np.arange(200), 5))
    t = pd.Series(pd.date_range("2021-01-01", periods=1000, freq="h"))
    tr, _ = grouped_split(groups, 0.2, seed=0)
    fit, val = validation_split(tr, t, "grouped", 0.2, seed=0, groups=groups)
    assert set(fit) | set(val) == set(tr)
    assert not set(groups.iloc[fit]) & set(groups.iloc[val])


def test_temporal_folds_roll_forward():
    t = pd.Series(pd.date_range("2020-01-01", periods=1000, freq="D")).sample(frac=1, random_state=0)
    t = t.reset_index(drop=True)
    folds = temporal_folds(t, 0.2, 3)
    assert [f.fold for f in folds] == [0, 1, 2]
    for f in folds:
        assert t.iloc[f.train].max() < t.iloc[f.test].min()
        assert abs(len(f.test) / 1000 - 0.2) < 0.01
    # expanding training window, consecutive non-overlapping test blocks
    assert len(folds[0].train) < len(folds[1].train) < len(folds[2].train)
    assert t.iloc[folds[0].test].max() < t.iloc[folds[1].test].min()
    # the last fold is the plain temporal split
    tr, te = temporal_split(t, 0.2)
    assert np.array_equal(folds[-1].train, tr) and np.array_equal(folds[-1].test, te)


def test_temporal_folds_reject_too_many():
    t = pd.Series(pd.date_range("2020-01-01", periods=100, freq="D"))
    try:
        temporal_folds(t, 0.2, 5)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_grouped_mode_needs_groups():
    t = pd.Series(pd.date_range("2020-01-01", periods=100, freq="D"))
    try:
        make_folds("grouped", t, None, 0.2, seed=0)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
