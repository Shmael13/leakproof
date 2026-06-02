"""Reproducible synthetic market data with *genuine* predictable structure.

The point of the benchmark is to measure how much in-sample Sharpe a leak
fabricates. For that we need data where (a) an honest backward-looking signal
earns a small, real edge, and (b) a leaking signal can fabricate a large one.

We build returns with two ingredients:
  * a persistent common factor (cross-sectional structure), and
  * mild AR(1) momentum in each asset's returns (a small, real, *backward*
    predictable component).

Everything is seeded, so `run_benchmark.py` is deterministic and needs no
network. Real-data variants (yfinance / ccxt) can be dropped in later behind
the same interface.
"""
import numpy as np


def make_panel(T=1500, A=30, seed=0):
    """Return (prices, fwd_returns) arrays of shape (T, A).

    fwd_returns[t] is the return realised from holding over (t -> t+1); it is
    accounting output, never a decision input.
    """
    rng = np.random.default_rng(seed)

    # weak persistent common factor (AR(1)) -> some cross-sectional structure
    factor = np.zeros(T)
    for t in range(1, T):
        factor[t] = 0.85 * factor[t - 1] + rng.normal(0, 0.0015)
    betas = rng.uniform(0.2, 0.8, size=A)

    # idiosyncratic AR(1) returns -> a *mild*, real, backward-looking edge
    eps = rng.normal(0, 0.015, size=(T, A))
    r = np.zeros((T, A))
    phi = 0.04                                   # small momentum coefficient
    for t in range(1, T):
        r[t] = phi * r[t - 1] + betas * factor[t] + eps[t]

    prices = 100 * np.exp(np.cumsum(r, axis=0))
    fwd = np.full((T, A), np.nan)
    fwd[:-1] = prices[1:] / prices[:-1] - 1.0
    return prices, fwd


def make_series(T=1500, seed=0):
    """Single-asset prices + forward returns, same generative process."""
    prices, fwd = make_panel(T=T, A=1, seed=seed)
    return prices[:, 0], fwd[:, 0]
