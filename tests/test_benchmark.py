"""Regression guard for the H3 benchmark: every buggy pattern must be rejected
and every correct pattern accepted, deterministically."""
from benchmarks.run_benchmark import collect


def test_all_leaks_caught_and_correct_accepted():
    rows = collect()
    assert rows, "no patterns ran"
    for row in rows:
        name, caught, accepted = row[0], row[2], row[3]
        assert caught, f"{name}: tni FAILED to reject the buggy pipeline"
        assert accepted, f"{name}: tni wrongly rejected the correct pipeline"


def test_determinism():
    a = collect()
    b = collect()
    assert a == b, "benchmark is not deterministic"
