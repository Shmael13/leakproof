"""Baselines the checker is meant to beat (DESIGN.md B.4).

Each baseline is a concrete, reproducible detector so the comparison is honest
rather than asserted:

  * linter      - a syntactic scan of the pipeline (the kind a ruff/flake8 rule
                  could do): flags forward shifts/leads, centered rolling, and
                  shuffled K-fold. No notion of *availability*, so it misses
                  every leak that is not a syntactic red flag.
  * unit_test   - a Sharpe-sanity test: flags a result whose in-sample Sharpe is
                  implausibly high (|Sharpe| > 5). Catches the egregious leaks,
                  sails past the subtle ones.
  * asof        - feature-store / point-in-time-join discipline: addresses
                  timestamp alignment and point-in-time universe membership, so
                  it catches publication-lag, survivorship and resampling, but
                  not statistical full-sample leakage or CV mistakes.

The point of the benchmark is that these catch *different, small* subsets, so
even their union misses leaks that tni rejects by construction.
"""

_SANITY_SHARPE = 5.0
# point-in-time / timestamp-alignment issues that ASOF-join discipline handles
_ASOF_CATCHES = {"publication-lag", "survivorship-universe",
                 "resampling-lookahead"}


def _ops(node):
    """All operator names in a node's provenance tree."""
    seen, out = set(), []

    def rec(nd):
        if id(nd) in seen:
            return
        seen.add(id(nd))
        out.append(getattr(nd, "_op", "") or "")
        for p in getattr(nd, "_parents", ()) or ():
            rec(p)

    rec(node)
    return out


def linter_catches(pattern):
    ops = " ".join(_ops(pattern.buggy))
    return ("shift(k=-" in ops) or ("center" in ops) or ("kfold" in ops)


def unit_test_catches(pattern, inflated_sharpe):
    return abs(inflated_sharpe) > _SANITY_SHARPE


def asof_catches(pattern):
    return pattern.name in _ASOF_CATCHES
