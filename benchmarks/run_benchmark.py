"""Run the leakage benchmark and print the headline table (H3).

For each pattern: does tni reject the buggy pipeline? accept the correct one?
and what in-sample Sharpe would the buggy pipeline have reported had the leak
gone undetected, vs. the honest pipeline?

Usage:  python benchmarks/run_benchmark.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tni.core import LeakageError                       # noqa: E402
from benchmarks import baselines, datasets, patterns     # noqa: E402

ANN = np.sqrt(252)


def _sharpe(pnl):
    pnl = pnl[np.isfinite(pnl)]
    if len(pnl) < 2 or pnl.std(ddof=1) == 0:
        return 0.0
    return float(pnl.mean() / pnl.std(ddof=1) * ANN)


def _pnl(values, ret, kind):
    if kind == "series":
        m = np.isfinite(values) & np.isfinite(ret)
        return np.where(m, values * ret, np.nan)
    pos = np.nan_to_num(values, nan=0.0)
    r = np.nan_to_num(ret, nan=0.0)
    return (pos * r).sum(axis=1)


def _rejects(expr):
    try:
        expr.to_positions(exec_lag=0)
        return False
    except LeakageError:
        return True


def evaluate(p):
    caught = _rejects(p.buggy)
    accepted = not _rejects(p.correct)
    sharpe_buggy = _sharpe(_pnl(p.buggy.values, p.buggy_ret, p.kind))
    sharpe_correct = _sharpe(_pnl(p.correct.values, p.correct_ret, p.kind))
    base = (baselines.linter_catches(p),
            baselines.unit_test_catches(p, sharpe_buggy),
            baselines.asof_catches(p))
    return caught, accepted, sharpe_buggy, sharpe_correct, base


def _load(source):
    """Return (prices_series, fwd_series, prices_panel, fwd_panel, label)."""
    if source == "synthetic":
        ps, fs = datasets.make_series(T=1500, seed=0)
        pp, fp = datasets.make_panel(T=1500, A=30, seed=0)
        return ps, fs, pp, fp, "synthetic (seeded)"
    from benchmarks import realdata
    if source == "equities":
        pp, fp, cols, _ = realdata.load_equities()
        j = cols.index("AAPL") if "AAPL" in cols else 0
    elif source == "crypto":
        pp, fp, cols, _ = realdata.load_crypto()
        j = cols.index("BTC/USD") if "BTC/USD" in cols else 0
    else:
        raise ValueError(f"unknown source: {source}")
    return pp[:, j], fp[:, j], pp, fp, f"{source}: {cols[j]} + {len(cols)}-asset panel"


def collect(source="synthetic"):
    """Run every pattern on the given data source; return result tuples."""
    prices_s, fwd_s, prices_p, fwd_p, _ = _load(source)
    rows = []
    for fn in patterns.SERIES_PATTERNS:
        rows.append(evaluate_named(fn(prices_s, fwd_s)))
    for fn in patterns.PANEL_PATTERNS:
        rows.append(evaluate_named(fn(prices_p, fwd_p)))
    return rows


def _mark(b):
    return "✓" if b else "·"


def render(rows):
    hdr = ("| Leakage pattern | tni | linter | unit-test | ASOF | "
           "Sharpe if undetected | honest |")
    sep = "|---|:--:|:--:|:--:|:--:|--:|--:|"
    lines = [hdr, sep]
    tot = {"tni": 0, "lint": 0, "unit": 0, "asof": 0, "acc": 0}
    for name, kind, caught, accepted, sb, sc, base in rows:
        lint, unit, asof = base
        tot["tni"] += caught
        tot["lint"] += lint
        tot["unit"] += unit
        tot["asof"] += asof
        tot["acc"] += accepted
        fp = "" if accepted else " ⚠FP"
        lines.append(f"| {name} | {_mark(caught)}{fp} | {_mark(lint)} "
                     f"| {_mark(unit)} | {_mark(asof)} | {sb:+.2f} | {sc:+.2f} |")
    n = len(rows)
    lines += [
        "",
        f"Leaks rejected: **tni {tot['tni']}/{n}**, linter {tot['lint']}/{n}, "
        f"unit-test {tot['unit']}/{n}, ASOF {tot['asof']}/{n}.",
        f"Correct pipelines accepted by tni (no false positive): {tot['acc']}/{n}.",
        "",
        "'Sharpe if undetected' is what the buggy pipeline would have reported "
        "had the leak slipped past review; tni rejects it at check time, before "
        "any backtest runs. ✓ = caught, · = missed.",
    ]
    return "\n".join(lines)


def evaluate_named(p):
    caught, accepted, sb, sc, base = evaluate(p)
    return (p.name, p.kind, caught, accepted, sb, sc, base)


def main():
    import sys
    sources = sys.argv[1:] or ["synthetic", "equities", "crypto"]
    sections = []
    for source in sources:
        try:
            _, _, _, _, label = _load(source)
            rows = collect(source)
        except Exception as e:                       # real-data fetch may fail
            print(f"[skip {source}: {type(e).__name__}: {e}]")
            continue
        table = render(rows)
        print(f"\n### {label}\n\n{table}")
        sections.append(f"## {label}\n\n{table}\n")

    out = os.path.join(os.path.dirname(__file__), "RESULTS.md")
    with open(out, "w") as f:
        f.write("# Leakage benchmark results (H3)\n\n")
        f.write("Regenerate with `python benchmarks/run_benchmark.py`. "
                "Real-data panels are cached under `data/` after the first fetch.\n\n")
        f.write("\n".join(sections))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
