"""Tests for the unsound AST linter (tni/ast_lint)."""
from tni.ast_lint.linter import lint_source

BUGGY = """
import pandas as pd
ret = df['close'].pct_change()
smooth = ret.rolling(11, center=True).mean()
z = (smooth - smooth.mean()) / smooth.std()
signal = z.shift(-1)
"""

CLEAN = """
import pandas as pd
ret = df['close'].pct_change()
smooth = ret.rolling(11).mean()
z = (smooth - smooth.rolling(60).mean()) / smooth.rolling(60).std()
signal = z.shift(1)
"""


def _codes(src):
    return {f.code for f in lint_source(src)}


def test_buggy_signatures_detected():
    codes = _codes(BUGGY)
    assert {"TNI-CENTER", "TNI-FULLSTAT", "TNI-LEAD"} <= codes


def test_clean_has_no_findings():
    assert _codes(CLEAN) == set()


def test_shuffled_cv_detected():
    src = ("from sklearn.model_selection import KFold\n"
           "kf = KFold(n_splits=5, shuffle=True)\n")
    assert "TNI-CV" in _codes(src)


def test_windowed_reduction_not_flagged():
    # a trailing-window mean must NOT be flagged as a full-sample statistic
    src = "z = (x - x.rolling(20).mean()) / x.rolling(20).std()\n"
    assert "TNI-FULLSTAT" not in _codes(src)
