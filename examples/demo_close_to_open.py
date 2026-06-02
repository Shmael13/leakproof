"""Day-0 spike: catch the close-to-open look-ahead leak end-to-end.

Run:  python examples/demo_close_to_open.py
No network / external data (synthetic random-walk prices with no real edge).
"""
import os
import sys

import numpy as np

from tni import LeakageError, TemporalSeries, backtest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    rng = np.random.default_rng(7)
    n = 1500
    # synthetic daily closes: geometric random walk -> NO genuine predictability
    daily = rng.normal(0, 0.01, size=n)
    close_prices = 100 * np.exp(np.cumsum(daily))

    # A daily CLOSE is knowable only at the END of its bar -> available at the
    # next decision point (avail_lag=1). Decisions for bar s must use info <= s.
    close = TemporalSeries.observe(close_prices, avail_lag=1, label="close")
    daily_ret = close.pct_change(label="daily_ret")

    print("=" * 72)
    print("BUGGY: trade bar s using bar s's OWN return (peeks at close[s])")
    print("=" * 72)
    leaky_signal = daily_ret.sign()  # decision for bar s uses ret[s]  -> leak
    try:
        leaky_signal.to_positions(exec_lag=0)
        print("  checker PASSED (unexpected!)")
    except LeakageError as e:
        print("  checker REJECTED at build time, before any backtest:")
        print("     ", e)
    inflated = backtest.run(leaky_signal, daily_ret)
    print(f"\n  (had the leak slipped through -> inflated Sharpe = "
          f"{inflated['sharpe']:.2f})")

    print("\n" + "=" * 72)
    print("CORRECT: trade bar s using YESTERDAY's return (momentum)")
    print("=" * 72)
    honest_signal = daily_ret.shift(1).sign()  # decision for bar s uses ret[s-1]
    positions = honest_signal.to_positions(exec_lag=0)  # type-checks
    honest = backtest.run(positions, daily_ret)
    print("  checker PASSED.")
    print(f"  honest Sharpe = {honest['sharpe']:.2f}  (n={honest['n']})")

    print("\nThe checker caught the look-ahead bias statically; the 'edge' it")
    print("removed was the entire inflated Sharpe.")


if __name__ == "__main__":
    main()
