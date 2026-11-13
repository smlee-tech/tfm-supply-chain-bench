import numpy as np

from tfmbench.metrics import bootstrap_ci, expected_calibration_error


def test_ece_near_zero_when_calibrated():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=200_000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    assert expected_calibration_error(y, p) < 0.01


def test_ece_large_when_overconfident():
    y = np.array([0, 1] * 500)
    p = np.where(y == 1, 0.6, 0.4)  # right direction, but 0.4/0.6 instead of 0/1
    assert abs(expected_calibration_error(y, p) - 0.4) < 1e-9


def test_bootstrap_ci_brackets_mean():
    y = np.arange(100, dtype=float)
    lo, hi = bootstrap_ci(y, y, lambda a, b: float(np.mean(a)), n_boot=300)
    assert lo < 49.5 < hi
