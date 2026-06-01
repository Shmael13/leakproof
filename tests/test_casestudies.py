"""H2 guardrail (no network): the correct strategy *shapes* must type-check.

Mirrors the three case studies on seeded synthetic data so the zero-false-
positive claim is checked without fetching real data.
"""
import numpy as np
from sklearn.linear_model import Ridge

from tni import TemporalFrame, TemporalSeries, available_at
from tni.model import make_lagged_features, predict_walkforward
from benchmarks import datasets


def test_cross_sectional_momentum_typechecks():
    prices, _ = datasets.make_panel(T=400, A=10, seed=1)
    px = TemporalFrame.observe(prices, avail_lag=0, label="price")
    sig = px.pct_change().rolling_back(20, "mean").shift(1).zscore_xs().demean_xs()
    sig.to_positions(exec_lag=0)            # must not raise


def test_pit_value_with_publication_lag_typechecks():
    prices, _ = datasets.make_panel(T=600, A=10, seed=2)
    px = TemporalFrame.observe(prices, avail_lag=0, label="price")
    long_ret = px.pct_change().rolling_back(120, "mean")
    published = available_at(long_ret, long_ret.avail + 45, reason="pub lag")
    sig = published.shift(45).scale(-1.0).zscore_xs().demean_xs()
    sig.to_positions(exec_lag=0)


def test_walkforward_ml_typechecks():
    prices, _ = datasets.make_series(T=500, seed=3)
    X, y, xa, ya = make_lagged_features(prices)
    pred = predict_walkforward(Ridge(), X, y, xa, ya, min_train=150,
                               retrain_every=20, embargo=5)
    pred.sign().to_positions(exec_lag=0)
