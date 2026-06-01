"""Unit tests for the availability transfer functions (DESIGN.md A.4).

Each test pins down "label in -> label out" for one operator, plus the
legality check in to_positions. A clean observation has avail[s] == s; an
operator is SAFE iff it keeps avail[s] <= s, and LEAKs iff it can exceed s.
"""
import numpy as np
import pytest

from tni.core import BOT, TOP, LeakageError
from tni.series import TemporalSeries

N = 20


def clean(label="x"):
    """A clean backward-looking series: value == index, avail[s] == s."""
    v = np.arange(N, dtype=float)
    return TemporalSeries(v, np.arange(N, dtype=float), label=label)


# --- sources / declassification -------------------------------------------
def test_observe_lag_zero_is_clean():
    s = TemporalSeries.observe(np.arange(N), avail_lag=0)
    assert np.array_equal(s.avail, np.arange(N))


def test_observe_lag_shifts_availability_forward():
    s = TemporalSeries.observe(np.arange(N), avail_lag=1)
    assert np.array_equal(s.avail, np.arange(N) + 1)


def test_constant_is_bottom():
    s = TemporalSeries.constant(3.0, N)
    assert np.all(s.avail == BOT)


# --- shift / lag / lead -----------------------------------------------------
def test_backward_lag_is_safe():
    s = clean().shift(1)            # value at s is x[s-1], avail = s-1
    assert s.avail[1] == 0 and s.avail[5] == 4


def test_forward_lead_leaks():
    s = clean().shift(-1)          # value at s is x[s+1], avail = s+1 > s
    assert s.avail[5] == 6
    with pytest.raises(LeakageError):
        s.to_positions(exec_lag=0)


# --- elementwise binops: label = join of operands ---------------------------
def test_binop_takes_join_of_labels():
    a = clean("a")
    b = clean("b").shift(-2)       # avail[s] = s+2
    out = a + b
    assert out.avail[5] == 7       # max(5, 7)


def test_pct_change_join():
    s = TemporalSeries.observe(np.arange(1, N + 1), avail_lag=1)
    r = s.pct_change()             # uses s and s-1, both avail (s+1, s) -> s+1
    assert r.avail[5] == 6


# --- rolling windows --------------------------------------------------------
def test_rolling_back_is_safe():
    s = clean().rolling(5, "back").mean()
    assert np.all(s.avail[4:] == np.arange(4, N))   # A(s) == s


def test_rolling_center_leaks():
    s = clean().rolling(5, "center").mean()
    assert s.avail[5] == 7                           # window reaches s+2
    with pytest.raises(LeakageError):
        s.to_positions(exec_lag=0)


def test_rolling_forward_leaks():
    s = clean().rolling(5, "forward").mean()
    assert s.avail[5] == 9
    with pytest.raises(LeakageError):
        s.to_positions(exec_lag=0)


def test_expanding_is_safe():
    s = clean().expanding().mean()
    assert np.all(s.avail == np.arange(N))           # join over [0..s] == s


# --- full-sample reductions -> TOP -> leak on use ---------------------------
def test_full_reduction_is_top():
    s = clean().full_mean()
    assert np.all(s.avail == TOP)


def test_zscore_full_leaks():
    s = clean().zscore_full()
    with pytest.raises(LeakageError):
        s.to_positions(exec_lag=0)


def test_zscore_back_is_safe():
    s = clean().zscore_back(5)
    # first window-1 rows are nan (no position); the rest must type-check
    s.to_positions(exec_lag=0)


def test_clip_with_full_sample_bounds_leaks():
    x = clean()
    s = x.clip(x.full_quantile(0.05), x.full_quantile(0.95))
    with pytest.raises(LeakageError):
        s.to_positions(exec_lag=0)


# --- exec lag ---------------------------------------------------------------
def test_exec_lag_requires_strictly_older_info():
    # a clean series has avail[s] == s; with exec_lag=1 we require avail <= s-1,
    # so even "today's" info is now illegal -> must lag by one bar first.
    clean().shift(1).to_positions(exec_lag=1)        # ok: avail[s] = s-1
    with pytest.raises(LeakageError):
        clean().to_positions(exec_lag=1)             # avail[s] = s > s-1
