# AST linter results (best-effort, UNSOUND)

This linter pattern-matches syntactic red flags in ordinary pandas/sklearn code. It has false positives and false negatives by design; it does not affect the soundness theorem. Ground truth is taken from the `_buggy`/`_clean` filename convention.


**cv_pipeline_buggy.py** (buggy) — 4 finding(s):
  - L11 [TNI-LEAD] negative .shift() reads future rows
  - L14 [TNI-FITALL] .fit_transform() likely fits a scaler on the whole sample (normalise after the train split)
  - L17 [TNI-CV] train_test_split shuffles by default (pass shuffle=False for time series)
  - L20 [TNI-CV] KFold(shuffle=True) on time series leaks

**ml_clean.py** (clean) — 1 finding(s):
  - L10 [TNI-LEAD] negative .shift() reads future rows

**signal_buggy.py** (buggy) — 4 finding(s):
  - L8 [TNI-CENTER] .rolling(center=True) peeks ahead
  - L11 [TNI-FULLSTAT] full-sample .std() in arithmetic (use a trailing window)
  - L11 [TNI-FULLSTAT] full-sample .mean() in arithmetic (use a trailing window)
  - L14 [TNI-LEAD] negative .shift() reads future rows

**signal_clean.py** (clean) — no findings

Recall on buggy scripts: 2/2 flagged.
False positives among clean scripts: 1/2 flagged.

The false positive is instructive: the linter flags any negative `.shift()`, but a forward-shifted *target* (y = next return) is legitimate -- a syntactic tool cannot tell a leaked feature from a label. The sound tni core has no such false positive (cf. H2: 3/3 correct strategies type-check), which is exactly why the core, not the linter, carries the guarantee.
