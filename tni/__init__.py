"""tni — temporal non-interference checker.

Sound embedded-DSL core: build a typed dataflow, type-check availability
labels, then run numerics. Look-ahead bias is rejected at check time.
"""
from . import backtest
from .core import BOT, TOP, LeakageError
from .declassify import audit_log, available_at, clear_audit
from .frame import TemporalFrame
from .series import Rolling, TemporalSeries

__all__ = ["TemporalSeries", "TemporalFrame", "Rolling",
           "LeakageError", "backtest", "BOT", "TOP",
           "available_at", "audit_log", "clear_audit"]
