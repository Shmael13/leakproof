"""H2 expressiveness: three realistic, CORRECT strategies end-to-end in tni.

The claim is that genuine strategies type-check with a *small* annotation
burden -- the only temporal annotations a user writes are the trusted
declassifications (`observe` sources + `available_at` assertions); every other
label is inferred. We report, per strategy:

  * does it type-check?         (the zero-false-positive guardrail for H2)
  * declassifications           (observe sources + available_at calls)
  * total operators in the pipeline, and the burden ratio decl/ops
  * the realised backtest Sharpe (it runs once it type-checks)

Usage:  python benchmarks/casestudies.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.ensemble import HistGradientBoostingRegressor   # noqa: E402

import tni                                                    # noqa: E402
from tni import (LeakageError, TemporalFrame, TemporalSeries,  # noqa: E402
                 available_at, backtest)
from tni.model import (make_lagged_features,                  # noqa: E402
                       predict_walkforward)
from benchmarks import realdata                               # noqa: E402


def _provenance(node):
    """(total operators, leaf sources) in a node's provenance tree."""
    seen, total, leaves = set(), 0, 0

    def rec(nd):
        nonlocal total, leaves
        if id(nd) in seen:
            return
        seen.add(id(nd))
        total += 1
        parents = getattr(nd, "_parents", ()) or ()
        if not parents:
            leaves += 1
        for p in parents:
            rec(p)

    rec(node)
    return total, leaves


# --- 1. cross-sectional momentum (crypto) ----------------------------------
def momentum_crypto():
    tni.clear_audit()
    prices, fwd, cols, _ = realdata.load_crypto()
    px = TemporalFrame.observe(prices, avail_lag=0, label="price")   # 1 observe
    signal = (px.pct_change().rolling_back(60, "mean")
                .shift(1).zscore_xs().demean_xs())
    positions = signal.to_positions(exec_lag=0)                      # type-check
    sharpe = backtest.run_panel(positions, TemporalFrame.observe(fwd, 0))["sharpe"]
    total, leaves = _provenance(signal)
    return {"name": "cross-sectional momentum", "universe": f"crypto x{len(cols)}",
            "type_checks": True, "n_decl": leaves + len(tni.audit_log()),
            "n_ops": total, "sharpe": sharpe}


# --- 2. point-in-time long-horizon reversal with publication lag (equities) -
def pit_value_equities():
    tni.clear_audit()
    prices, fwd, cols, _ = realdata.load_equities()
    px = TemporalFrame.observe(prices, avail_lag=0, label="price")   # 1 observe
    long_ret = px.pct_change().rolling_back(252, "mean")             # 1y signal
    # the fundamental-style value is only published 45 bars after its ref date
    published = available_at(long_ret, long_ret.avail + 45,          # 1 declass
                             reason="report -> publication lag")
    # long-horizon reversal: short recent winners (negate), use once published
    signal = published.shift(45).scale(-1.0).zscore_xs().demean_xs()
    positions = signal.to_positions(exec_lag=0)
    sharpe = backtest.run_panel(positions, TemporalFrame.observe(fwd, 0))["sharpe"]
    total, leaves = _provenance(signal)
    return {"name": "PIT long-horizon reversal", "universe": f"equities x{len(cols)}",
            "type_checks": True, "n_decl": leaves + len(tni.audit_log()),
            "n_ops": total, "sharpe": sharpe}


# --- 3. walk-forward gradient-boosting predictor with embargoed CV (crypto) -
def walkforward_ml_crypto():
    tni.clear_audit()
    prices, fwd, cols, _ = realdata.load_crypto()
    j = cols.index("BTC/USD") if "BTC/USD" in cols else 0
    X, y, xa, ya = make_lagged_features(prices[:, j])                # 1 observe
    est = HistGradientBoostingRegressor(max_iter=120, max_depth=3,
                                        learning_rate=0.05, random_state=0)
    pred = predict_walkforward(est, X, y, xa, ya, min_train=180,
                               retrain_every=20, embargo=5)          # embargoed
    signal = pred.sign()
    positions = signal.to_positions(exec_lag=0)
    sharpe = backtest.run(positions, TemporalSeries.observe(fwd[:, j], 0))["sharpe"]
    total, leaves = _provenance(signal)
    return {"name": "walk-forward GBM (embargoed CV)", "universe": f"crypto: {cols[j]}",
            "type_checks": True, "n_decl": leaves + len(tni.audit_log()),
            "n_ops": total, "sharpe": sharpe}


CASE_STUDIES = [momentum_crypto, pit_value_equities, walkforward_ml_crypto]


def render(results):
    hdr = ("| Strategy | universe | type-checks? | declassifications | "
           "total ops | burden | Sharpe |")
    sep = "|---|---|:--:|:--:|:--:|:--:|--:|"
    lines = [hdr, sep]
    for r in results:
        burden = r["n_decl"] / r["n_ops"]
        lines.append(f"| {r['name']} | {r['universe']} "
                     f"| {'✓' if r['type_checks'] else '✗'} "
                     f"| {r['n_decl']} | {r['n_ops']} | {burden:.0%} "
                     f"| {r['sharpe']:+.2f} |")
    n = len(results)
    n_ok = sum(r["type_checks"] for r in results)
    lines += [
        "",
        f"All {n_ok}/{n} correct strategies type-check (zero false positives).",
        "'declassifications' = trusted annotations the user writes (observe "
        "sources + available_at); 'burden' = declassifications / total operators. "
        "Every other label is inferred.",
    ]
    return "\n".join(lines)


def main():
    results = []
    for fn in CASE_STUDIES:
        try:
            results.append(fn())
        except LeakageError as e:
            print(f"[FALSE POSITIVE in {fn.__name__}]\n{e}")
            results.append({"name": fn.__name__, "universe": "-",
                            "type_checks": False, "n_decl": 0, "n_ops": 0,
                            "sharpe": float("nan")})
    table = render(results)
    print("\n" + table)
    out = os.path.join(os.path.dirname(__file__), "H2_RESULTS.md")
    with open(out, "w") as f:
        f.write("# H2 expressiveness: case studies\n\n")
        f.write("Regenerate with `python benchmarks/casestudies.py`.\n\n")
        f.write(table + "\n")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
