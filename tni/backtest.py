"""Minimal backtest accounting for the spike."""
import numpy as np


def sharpe(pnl, periods_per_year=252):
    pnl = np.asarray(pnl, dtype=float)
    pnl = pnl[np.isfinite(pnl)]
    if len(pnl) < 2 or pnl.std(ddof=1) == 0:
        return 0.0
    return float(pnl.mean() / pnl.std(ddof=1) * np.sqrt(periods_per_year))


def run(positions, returns, periods_per_year=252):
    """PnL of holding positions[s] over realized returns[s].

    `returns` is after-the-fact accounting, not a decision input, so it carries
    no availability constraint here.
    """
    pos = positions.values
    ret = returns.values
    mask = np.isfinite(pos) & np.isfinite(ret)
    pnl = np.where(mask, pos * ret, np.nan)
    return {
        "sharpe": sharpe(pnl[mask], periods_per_year),
        "pnl": pnl,
        "n": int(mask.sum()),
    }


def run_panel(positions, returns, periods_per_year=252):
    """Cross-sectional backtest: per-bar PnL is sum over assets of pos*ret.

    positions and returns are TemporalFrames of shape (T, A). Returns the
    portfolio PnL series and its Sharpe.
    """
    pos = np.nan_to_num(positions.values, nan=0.0)
    ret = np.nan_to_num(returns.values, nan=0.0)
    bar_pnl = (pos * ret).sum(axis=1)
    return {
        "sharpe": sharpe(bar_pnl, periods_per_year),
        "pnl": bar_pnl,
        "n": int(np.isfinite(bar_pnl).sum()),
    }
