"""Declassification: the trusted base (DESIGN.md A.5).

Everything else in tni is sound by construction -- labels are derived, never
asserted. Two primitives are the *only* places where trust is injected, i.e.
where a human vouches for an availability the system cannot derive:

  * `observe(stream, lag=d)`  -- a raw source's publication lag
    (already a constructor on TemporalSeries / TemporalFrame), and
  * `available_at(x, tau)`    -- assert that x is available at time tau.

`available_at` is the audited downgrade analogous to declassification in
information-flow control: it can hide a real leak, so every call is recorded in
an audit log. The selling point is that the trusted surface is small and
*enumerable*: `audit_log()` lists every assertion, so a reviewer checks those
instead of the whole pipeline.
"""
import numpy as np

from .frame import TemporalFrame
from .series import TemporalSeries

_AUDIT = []


def audit_log():
    """Return the list of declassifications made so far: (source, tau, reason)."""
    return list(_AUDIT)


def clear_audit():
    _AUDIT.clear()


def available_at(x, tau, reason=""):
    """Assert (on the user's authority) that `x` is available by time `tau`.

    `tau` may be a scalar (broadcast) or a per-row array. This OVERRIDES the
    inferred label, so it is trusted and logged. Use it for availabilities the
    calculus cannot derive -- e.g. a per-row point-in-time publication date.
    """
    _AUDIT.append((getattr(x, "label", "?"), tau, reason))
    if isinstance(x, TemporalSeries):
        n = len(x)
        avail = np.full(n, float(tau)) if np.isscalar(tau) else np.asarray(tau, float)
        return TemporalSeries(x.values, avail, label=f"available_at({x.label})",
                              parents=(x,), op=f"available_at(tau={_tau_str(tau)})")
    if isinstance(x, TemporalFrame):
        if np.isscalar(tau):
            avail = np.full_like(x.avail, float(tau))
        else:
            tau = np.asarray(tau, float)
            avail = tau if tau.ndim == 2 else np.broadcast_to(tau[:, None], x.shape).copy()
        return TemporalFrame(x.values, avail, label=f"available_at({x.label})",
                             parents=(x,), op=f"available_at(tau={_tau_str(tau)})")
    raise TypeError(f"available_at expects a TemporalSeries/Frame, got {type(x)}")


def _tau_str(tau):
    return "vec" if not np.isscalar(tau) else f"{tau:g}"
