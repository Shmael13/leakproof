"""The leakage benchmark: each pattern is a buggy/correct pair.

A Pattern packages two pipelines that share intent and differ only in a leak,
together with the realised return stream each position earns. The harness then
asks, for every pattern:
  * does tni REJECT the buggy pipeline at check time?      (utility)
  * does tni ACCEPT the correct pipeline?                  (no false positive)
  * what in-sample Sharpe would the buggy pipeline have reported had the leak
    gone undetected, vs. the honest pipeline?              (the money result)

All pipelines are built with the tni eDSL, so "rejection" means a LeakageError
raised before any numbers are produced.
"""
from dataclasses import dataclass

import numpy as np

from tni import TemporalFrame, TemporalSeries, available_at
from tni.core import TOP


@dataclass
class Pattern:
    name: str
    kind: str                # 'series' or 'panel'
    buggy: object            # TemporalSeries / TemporalFrame
    correct: object
    buggy_ret: np.ndarray    # return stream the buggy positions earn
    correct_ret: np.ndarray  # return stream the correct positions earn
    note: str = ""


# --- helpers ---------------------------------------------------------------
def _series_momentum(prices, window=20):
    """Honest backward momentum signal on a single asset (avail s)."""
    px = TemporalSeries.observe(prices, avail_lag=1, label="close")   # close@s+1
    ret = px.pct_change()                       # avail s+1
    mom = ret.rolling(window, "back").mean()    # avail s+1
    return px, ret, mom.shift(1)                # shift -> avail s (tradeable)


def _panel_momentum(prices, window=20):
    px = TemporalFrame.observe(prices, avail_lag=0, label="price")
    ret = px.pct_change()                       # avail s
    mom = ret.rolling_back(window, "mean")      # avail s
    return px, ret, mom.shift(1)                # avail s-1 (tradeable)


# --- SERIES patterns -------------------------------------------------------
def close_to_open(prices, fwd):
    px, ret, mom = _series_momentum(prices)
    # BUG: trade bar s on bar s's own (close-to-close) return -> not yet known
    buggy = ret.sign()
    correct = mom.sign()
    # the leak lets you capture the very move you peeked at (contemporaneous)
    contemp = np.full_like(prices, np.nan)
    contemp[1:] = prices[1:] / prices[:-1] - 1.0
    return Pattern("close-to-open", "series", buggy, correct,
                   buggy_ret=contemp, correct_ret=fwd,
                   note="trade at open using same-day close")


def target_leakage(prices, fwd):
    # BUG: feature is literally the forward return (shift(-1)) -> the label
    px = TemporalSeries.observe(prices, avail_lag=1, label="close")
    future_ret = px.pct_change().shift(-1)      # avail s+2: tomorrow's return
    buggy = future_ret.sign()
    _, _, mom = _series_momentum(prices)
    correct = mom.sign()
    return Pattern("target-leakage", "series", buggy, correct,
                   buggy_ret=fwd, correct_ret=fwd,
                   note="feature derived from the future label")


def full_sample_zscore(prices, fwd):
    _, ret, mom = _series_momentum(prices)
    buggy = mom.zscore_full()                   # mean/std over ALL time
    correct = mom.zscore_back(60)               # trailing standardisation
    return Pattern("full-sample-zscore", "series", buggy, correct,
                   buggy_ret=fwd, correct_ret=fwd,
                   note="standardise using whole-sample statistics")


def centered_rolling(prices, fwd):
    _, ret, mom = _series_momentum(prices)
    # BUG: smooth the signal with a CENTERED window (uses future bars)
    buggy = ret.rolling(11, "center").mean().sign()
    correct = ret.rolling(11, "back").mean().shift(1).sign()
    return Pattern("centered-rolling", "series", buggy, correct,
                   buggy_ret=fwd, correct_ret=fwd,
                   note="centered smoothing peeks ahead")


def winsorize_full_sample(prices, fwd):
    _, ret, mom = _series_momentum(prices)
    lo, hi = mom.full_quantile(0.05), mom.full_quantile(0.95)
    buggy = mom.clip(lo, hi).sign()             # bounds from the whole sample
    r = mom.rolling(120, "back")
    correct = mom.clip(r.quantile(0.05), r.quantile(0.95)).sign()
    return Pattern("winsorize-full-sample", "series", buggy, correct,
                   buggy_ret=fwd, correct_ret=fwd,
                   note="clip bounds computed over the whole sample")


# --- PANEL patterns --------------------------------------------------------
def panel_zscore(prices, fwd):
    px, ret, mom = _panel_momentum(prices)
    buggy = mom.zscore_panel().demean_xs()      # pool over all (time, asset)
    correct = mom.zscore_xs().demean_xs()       # per-date cross-section
    return Pattern("panel-zscore", "panel", buggy, correct,
                   buggy_ret=fwd, correct_ret=fwd,
                   note="pool normalisation over the whole panel")


def survivorship(prices, fwd):
    # An equal-weight long-only book over a universe. The bias is entirely in
    # *which universe*: trading only the names that turned out to be winners.
    T, A = prices.shape
    # CORRECT: point-in-time universe = every name (all listed from the start),
    # equal-weighted long book (membership knowable at s).
    correct = TemporalFrame.observe(np.full((T, A), 1.0 / A), avail_lag=0,
                                    label="universe_ew")

    # BUG: keep only full-sample winners (membership knowable only at the end).
    total_ret = prices[-1] / prices[0] - 1.0
    survivors = (total_ret >= np.median(total_ret)).astype(float)
    mask_vals = np.broadcast_to(survivors / survivors.sum(), prices.shape).copy()
    buggy = TemporalFrame(mask_vals, np.full_like(prices, TOP),
                          label="survivor_universe")
    return Pattern("survivorship-universe", "panel", buggy, correct,
                   buggy_ret=fwd, correct_ret=fwd,
                   note="universe = full-sample winners (delisted losers dropped)")


def target_leakage_panel(prices, fwd):
    px, ret, mom = _panel_momentum(prices)
    correct = mom.zscore_xs().demean_xs()
    future_ret = px.pct_change().shift(-1)      # tomorrow's return as a feature
    buggy = future_ret.zscore_xs().demean_xs()
    return Pattern("target-leakage-panel", "panel", buggy, correct,
                   buggy_ret=fwd, correct_ret=fwd,
                   note="cross-sectional signal = future return")


def resampling_lookahead(prices, fwd):
    _, ret, mom = _series_momentum(prices)
    # BUG: use this week's mean return (a period-end aggregate) intraperiod
    buggy = ret.resample(5, "mean", lag_periods=0).sign()
    correct = ret.resample(5, "mean", lag_periods=1).sign()
    return Pattern("resampling-lookahead", "series", buggy, correct,
                   buggy_ret=fwd, correct_ret=fwd,
                   note="period-end aggregate used within the period")


def global_feature_selection(prices, fwd):
    # weight several lagged-return features by their correlation with the target
    _, ret, _ = _series_momentum(prices)
    feats = [ret.rolling(w, "back").mean().shift(1) for w in (5, 10, 20, 60)]
    fwd_s = TemporalSeries.observe(np.concatenate([[np.nan], fwd[:-1]]),
                                   avail_lag=1, label="fwd")  # target, avail s+1

    def weighted(weight_of):
        out = None
        for f in feats:
            term = f * weight_of(f)
            out = term if out is None else out + term
        return out.sign()

    # BUG: weights from FULL-SAMPLE feature-target correlation (uses the future)
    buggy = weighted(lambda f: (f * fwd_s).full_mean())
    # correct: weights from an expanding (backward-only) correlation
    correct = weighted(lambda f: (f * fwd_s).expanding().mean().shift(1))
    return Pattern("global-feature-selection", "series", buggy, correct,
                   buggy_ret=fwd, correct_ret=fwd,
                   note="feature weights from whole-sample target correlation")


def publication_lag(prices, fwd):
    # a 'fundamental' (slow signal) published L bars after its reference date
    L = 5
    px = TemporalSeries.observe(prices, avail_lag=0, label="close")
    fundamental = px.pct_change().rolling(63, "back").mean()    # avail s
    published = available_at(fundamental, fundamental.avail + L,
                             reason="report -> publication lag")  # avail s+L
    # BUG: use the fundamental dated s at time s, though it is not published
    # until s+L.
    buggy = published.sign()
    # correct: at time s use the most recent *published* fundamental (dated s-L)
    correct = published.shift(L).sign()
    return Pattern("publication-lag", "series", buggy, correct,
                   buggy_ret=fwd, correct_ret=fwd,
                   note="fundamental used at report date, before publication")


def cv_leakage(prices, fwd):
    """Improper cross-validation: shuffled K-fold vs. walk-forward."""
    from sklearn.linear_model import Ridge

    from tni.model import (make_lagged_features, predict_kfold,
                           predict_walkforward)
    X, y, xa, ya = make_lagged_features(prices)
    est = Ridge(alpha=1.0)
    buggy = predict_kfold(est, X, y, xa, ya, n_splits=5,
                          shuffle=True, seed=0).sign()
    correct = predict_walkforward(est, X, y, xa, ya,
                                  min_train=200, retrain_every=20).sign()
    return Pattern("shuffled-kfold-cv", "series", buggy, correct,
                   buggy_ret=fwd, correct_ret=fwd,
                   note="shuffled K-fold trains on future folds")


SERIES_PATTERNS = [close_to_open, target_leakage, full_sample_zscore,
                   centered_rolling, winsorize_full_sample, cv_leakage,
                   resampling_lookahead, global_feature_selection,
                   publication_lag]
PANEL_PATTERNS = [panel_zscore, survivorship, target_leakage_panel]
