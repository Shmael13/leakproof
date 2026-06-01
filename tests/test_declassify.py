"""available_at() is the audited trusted downgrade (DESIGN.md A.5)."""
import numpy as np
import pytest

from tni import (LeakageError, TemporalSeries, audit_log, available_at,
                 clear_audit)


def _clean(n=20):
    return TemporalSeries(np.arange(n, dtype=float), np.arange(n, dtype=float), label="x")


def test_available_at_overrides_label_and_is_audited():
    clear_audit()
    leaky = _clean().shift(-1)                 # avail[s] = s+1 -> would leak
    with pytest.raises(LeakageError):
        leaky.to_positions(exec_lag=0)

    # the user asserts (on their authority) it is actually available at s
    fixed = available_at(leaky, np.arange(20, dtype=float))
    fixed.to_positions(exec_lag=0)             # now type-checks

    log = audit_log()
    assert len(log) == 1 and log[0][0] == leaky.label


def test_per_row_publication_lag():
    # fundamentals reported at row s but only published 3 rows later
    raw = _clean()
    pub = available_at(raw, raw.avail + 3, reason="report->publication lag")
    assert np.all(pub.avail == raw.avail + 3)
    with pytest.raises(LeakageError):
        pub.to_positions(exec_lag=0)           # published late -> illegal at s


def test_audit_log_is_enumerable():
    clear_audit()
    x = _clean()
    available_at(x, 0.0, reason="a")
    available_at(x, 1.0, reason="b")
    assert [r[2] for r in audit_log()] == ["a", "b"]
