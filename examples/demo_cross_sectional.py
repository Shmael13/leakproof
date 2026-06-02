"""Cross-sectional momentum: the full-sample preprocessing leak.

Two pipelines, identical except for ONE line:
  buggy   - standardise the signal with PANEL (full-sample) z-score  -> LEAK
  correct - standardise the signal CROSS-SECTIONALLY per date        -> SAFE

The only difference is whether the normalising statistics peek at the whole
sample (including the future). tni rejects the buggy one before any backtest.

Run:  python examples/demo_cross_sectional.py   (synthetic data, no network)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tni import LeakageError, TemporalFrame, backtest


def main():
    rng = np.random.default_rng(3)
    T, A = 1200, 30
    # synthetic asset prices: independent geometric random walks (no real edge)
    shocks = rng.normal(0, 0.012, size=(T, A))
    prices = 100 * np.exp(np.cumsum(shocks, axis=0))

    px = TemporalFrame.observe(prices, avail_lag=0, label="price")
    ret = px.pct_change()                       # return into day s, avail s
    mom = ret.rolling_back(20, "mean")          # 20-day trailing mean, avail s
    signal = mom.shift(1)                        # decide on info through s-1

    # forward returns are *accounting*, not a decision input (no constraint)
    fwd = np.full_like(prices, np.nan)
    fwd[:-1] = prices[1:] / prices[:-1] - 1.0
    fwd_frame = TemporalFrame.observe(fwd, avail_lag=0)

    print("=" * 72)
    print("BUGGY: standardise the signal with a FULL-SAMPLE (panel) z-score")
    print("=" * 72)
    buggy = signal.zscore_panel()                # mean/std over ALL time -> leak
    try:
        buggy.to_positions(exec_lag=0)
        print("  checker PASSED (unexpected!)")
    except LeakageError as e:
        print("  checker REJECTED at build time, before any backtest:")
        print("     ", e)
    inflated = backtest.run_panel(buggy, fwd_frame)
    print(f"\n  (had the leak slipped through -> inflated Sharpe = "
          f"{inflated['sharpe']:.2f})")

    print("\n" + "=" * 72)
    print("CORRECT: standardise CROSS-SECTIONALLY (per date), demeaned weights")
    print("=" * 72)
    honest = signal.zscore_xs().demean_xs()      # per-date, uses only date s-1
    positions = honest.to_positions(exec_lag=0)  # type-checks
    res = backtest.run_panel(positions, fwd_frame)
    print("  checker PASSED.")
    print(f"  honest Sharpe = {res['sharpe']:.2f}  (bars={res['n']})")

    print("\nThe leak was entirely in the normalisation touching the future;")
    print("tni localises it to the zscore_panel op, before a single PnL is computed.")


if __name__ == "__main__":
    main()
