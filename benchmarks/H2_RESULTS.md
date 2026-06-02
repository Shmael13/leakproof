# H2 expressiveness: case studies

Regenerate with `python benchmarks/casestudies.py`.

| Strategy | universe | type-checks? | declassifications | total ops | burden | Sharpe |
|---|---|:--:|:--:|:--:|:--:|--:|
| cross-sectional momentum | crypto x8 | ✓ | 1 | 6 | 17% | +0.33 |
| PIT long-horizon reversal | equities x20 | ✓ | 2 | 8 | 25% | -0.16 |
| walk-forward GBM (embargoed CV) | crypto: BTC/USD | ✓ | 1 | 2 | 50% | +1.00 |

All 3/3 correct strategies type-check (zero false positives).
'declassifications' = trusted annotations the user writes (observe sources + available_at); 'burden' = declassifications / total operators. Every other label is inferred.
