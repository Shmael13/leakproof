"""H1 (soundness), checked empirically: well-typed => temporally non-interferent.

Drives thousands of random pipelines through the perturb-the-future oracle and
asserts no pipeline both type-checks and lets a future input change a past
decision. Also asserts the test is non-vacuous (many pipelines genuinely
interfere, so the checker is actually doing work).
"""
import functools

from tni.testing import run_campaign


@functools.lru_cache(maxsize=1)
def _campaign():
    return run_campaign(n_pipelines=3000, n=40, seed=0)


def test_no_soundness_violations():
    s = _campaign()
    assert s["violations"] == [], (
        f"{len(s['violations'])} pipelines type-checked yet depend on the "
        f"future: {s['violations'][:3]}"
    )


def test_oracle_is_non_vacuous():
    # If nothing interfered, the property would hold trivially. We require both
    # that real leaks exist and that the checker rejects all of them.
    s = _campaign()
    assert s["interfering"] > 100, "oracle never observed interference"
    assert s["type_checked"] > 100, "no pipeline ever type-checked"


def test_seed_independence():
    for seed in (1, 7, 42):
        s = run_campaign(n_pipelines=800, n=32, seed=seed)
        assert s["violations"] == []
