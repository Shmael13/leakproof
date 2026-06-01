"""Tests for TemporalFrame: temporal ops down time, cross-sectional ops across
assets (safe), and panel-pooled ops (leak)."""
import numpy as np
import pytest

from tni.core import TOP, LeakageError
from tni.frame import TemporalFrame

T, A = 12, 4


def clean_panel(lag=0):
    v = np.arange(T * A, dtype=float).reshape(T, A)
    return TemporalFrame.observe(v, avail_lag=lag, label="px")


def test_observe_row_availability():
    f = clean_panel(lag=1)
    assert np.all(f.avail[5] == 6)        # every asset in row 5 avail at 6


def test_shift_back_is_safe():
    f = clean_panel().shift(1)
    assert np.all(f.avail[5] == 4)
    f.to_positions(exec_lag=0)


def test_shift_forward_leaks():
    f = clean_panel().shift(-1)
    with pytest.raises(LeakageError):
        f.to_positions(exec_lag=0)


def test_cross_sectional_rank_is_safe():
    # rank across assets at each date uses only same-date data -> A(s,.) == s
    f = clean_panel().rank_xs()
    assert np.all(f.avail[5] == 5)
    f.to_positions(exec_lag=0)
    # ranks lie in [0, 1]
    assert np.nanmin(f.values) >= 0 and np.nanmax(f.values) <= 1


def test_cross_sectional_zscore_is_safe():
    f = clean_panel().zscore_xs()
    f.to_positions(exec_lag=0)


def test_panel_pooled_zscore_leaks():
    f = clean_panel().zscore_panel()
    assert np.all(f.avail == TOP)
    with pytest.raises(LeakageError):
        f.to_positions(exec_lag=0)


def test_rolling_back_then_xs_momentum_is_safe():
    # a realistic cross-sectional momentum signal: rank of trailing return
    px = clean_panel(lag=1)
    mom = px.pct_change().rolling_back(3, "mean")   # trailing mean return
    signal = mom.shift(1).rank_xs()                 # decide on lagged, ranked
    signal.to_positions(exec_lag=0)                 # must type-check


def test_xs_zscore_preserves_lagged_availability():
    # if inputs are published with a lag, the cross-section inherits that lag
    f = clean_panel(lag=1).zscore_xs()
    assert np.all(f.avail[5] == 6)
    with pytest.raises(LeakageError):
        f.to_positions(exec_lag=0)                  # avail 6 > required 5
