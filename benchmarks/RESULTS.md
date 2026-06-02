# Leakage benchmark results (H3)

Regenerate with `python benchmarks/run_benchmark.py`. Real-data panels are cached under `data/` after the first fetch.

## crypto: BTC/USD + 8-asset panel

| Leakage pattern | tni | linter | unit-test | ASOF | Sharpe if undetected | honest |
|---|:--:|:--:|:--:|:--:|--:|--:|
| close-to-open | ✓ | · | ✓ | · | +15.89 | -0.39 |
| target-leakage | ✓ | ✓ | ✓ | · | +15.89 | -0.39 |
| full-sample-zscore | ✓ | · | · | · | -0.22 | -0.01 |
| centered-rolling | ✓ | ✓ | · | · | +3.22 | +0.24 |
| winsorize-full-sample | ✓ | · | · | · | -0.39 | -0.40 |
| shuffled-kfold-cv | ✓ | ✓ | · | · | +0.02 | -0.39 |
| resampling-lookahead | ✓ | · | · | ✓ | +4.23 | +0.12 |
| global-feature-selection | ✓ | · | · | · | +0.12 | -0.62 |
| publication-lag | ✓ | · | · | ✓ | +0.17 | +0.06 |
| panel-zscore | ✓ | · | · | · | +0.48 | +0.41 |
| survivorship-universe | ✓ | · | · | ✓ | +0.41 | +0.15 |
| target-leakage-panel | ✓ | ✓ | ✓ | · | +20.28 | +0.41 |

Leaks rejected: **tni 12/12**, linter 4/12, unit-test 3/12, ASOF 3/12.
Correct pipelines accepted by tni (no false positive): 12/12.

'Sharpe if undetected' is what the buggy pipeline would have reported had the leak slipped past review; tni rejects it at check time, before any backtest runs. ✓ = caught, · = missed.
