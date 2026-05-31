"""Soundness oracle: the perturb-the-future property test (H1).

This is an *independent* check on the type system. The type checker says a
pipeline is leakage-free by inspecting availability labels. The oracle says so
by experiment: perturb every input datum that is only knowable *after* a cut
time s*, and verify that no decision dated <= s* changes. If a pipeline
type-checks but the oracle finds a changed past decision, a transfer function
is unsound -- a bug to fix.

A pipeline is represented as a builder `f(raw_values) -> TemporalSeries` plus
the publication lag `delta` of its single raw input, so it can be rerun on a
perturbed dataset.
"""
import numpy as np

from .core import LeakageError
from .series import TemporalSeries


# --- operator moves ---------------------------------------------------------
# Each move maps a TemporalSeries -> TemporalSeries. SAFE moves never push a
# clean label above s; UNSAFE moves can.
def _safe_moves(rng):
    w = int(rng.integers(2, 8))
    k = int(rng.integers(1, 5))
    return [
        lambda x: x.shift(k),
        lambda x: x.rolling(w, "back").mean(),
        lambda x: x.rolling(w, "back").std(),
        lambda x: x.rolling(w, "back").max(),
        lambda x: x.expanding().mean(),
        lambda x: x.pct_change(),
        lambda x: x.sign(),
        lambda x: x + x.shift(k),
        lambda x: x - x.rolling(w, "back").mean(),
        lambda x: x.zscore_back(w),
    ]


def _unsafe_moves(rng):
    w = int(rng.integers(3, 8))
    k = int(rng.integers(1, 5))
    return [
        lambda x: x.shift(-k),
        lambda x: x.rolling(w, "center").mean(),
        lambda x: x.rolling(w, "forward").mean(),
        lambda x: x.zscore_full(),
        lambda x: x - x.full_mean(),
    ]


def gen_pipeline(rng, depth, allow_unsafe):
    """Return (builder, delta). builder(raw) applies a random op chain."""
    delta = int(rng.integers(0, 2))   # publication lag 0 or 1
    ops = []
    for _ in range(depth):
        pool = _safe_moves(rng)
        if allow_unsafe and rng.random() < 0.5:
            pool = pool + _unsafe_moves(rng)
        ops.append(pool[int(rng.integers(len(pool)))])

    def builder(raw):
        x = TemporalSeries.observe(raw, avail_lag=delta, label="raw")
        for op in ops:
            x = op(x)
        return x

    return builder, delta


def type_checks(series, exec_lag=0):
    try:
        series.to_positions(exec_lag=exec_lag)
        return True
    except LeakageError:
        return False


def _eq(a, b):
    return (a == b) | (np.isnan(a) & np.isnan(b))


def perturb_check(builder, delta, raw, rng, exec_lag=0):
    """Run the oracle for one pipeline on one base dataset.

    Returns a dict:
      type_checks  - did the checker accept the pipeline?
      interferes   - did perturbing the future change any past decision?
      sound_ok     - the soundness obligation: not (type_checks and interferes)
    """
    n = len(raw)
    z1 = builder(raw)
    ok = type_checks(z1, exec_lag)

    s_star = int(rng.integers(0, n))
    # rows knowable only after s*: true availability j + delta > s*
    future = np.array([j for j in range(n) if j + delta > s_star], dtype=int)
    raw2 = np.array(raw, dtype=float)
    if len(future):
        raw2[future] = raw[future] + rng.normal(0, 5.0, size=len(future)) + 7.0
    z2 = builder(raw2)

    past = slice(0, s_star + 1)
    interferes = not np.all(_eq(z1.values[past], z2.values[past]))
    return {
        "type_checks": ok,
        "interferes": interferes,
        "sound_ok": not (ok and interferes),
        "s_star": s_star,
    }


def run_campaign(n_pipelines=2000, n=40, seed=0, allow_unsafe=True):
    """Drive many random pipelines through the oracle; return a summary."""
    rng = np.random.default_rng(seed)
    stats = {"total": 0, "type_checked": 0, "leaky": 0,
             "interfering": 0, "violations": []}
    for i in range(n_pipelines):
        depth = int(rng.integers(1, 5))
        builder, delta = gen_pipeline(rng, depth, allow_unsafe)
        raw = rng.normal(0, 1, size=n).cumsum() + 100.0
        r = perturb_check(builder, delta, raw, rng)
        stats["total"] += 1
        stats["type_checked"] += int(r["type_checks"])
        stats["leaky"] += int(not r["type_checks"])
        stats["interfering"] += int(r["interferes"])
        if not r["sound_ok"]:
            stats["violations"].append((i, r))
    return stats
