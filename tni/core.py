"""Core lattice + errors.

Availability labels live on the time lattice (T, <=) with join=max, meet=min:
  BOT (-inf) = "always available" (constants),
  TOP (+inf) = "never available within the sample" (full-sample / future).
"""
import numpy as np

BOT = -np.inf  # constants: available for all time
TOP = np.inf   # full-sample / future: never available in-sample


def _fmt(a):
    if a == TOP:
        return "TOP(+inf)"
    if a == BOT:
        return "BOT(-inf)"
    return f"{a:g}"


def _avail_at(node, s):
    """Decision-time availability of a node at time s (row-max for panels)."""
    a = node.avail
    return float(a[s]) if a.ndim == 1 else float(np.max(a[s]))


def explain_leak(node, s, required):
    """Render the dependency path that carries future information into time s.

    Walks the retained provenance (`_parents`), descending only through
    operands that are themselves illegal at s, and marks the node where the
    future information first enters (a flagged node with no flagged parent).
    """
    lines = []

    def rec(nd, depth):
        a = _avail_at(nd, s)
        parents = getattr(nd, "_parents", ()) or ()
        bad_parents = [p for p in parents if _avail_at(p, s) > required]
        marker = "   <== future information enters here" if not bad_parents else ""
        name = getattr(nd, "_op", None) or getattr(nd, "label", "?")
        lines.append(f"{'    ' * depth}{name}: A({s})={_fmt(a)}"
                     f" (need <= {required}){marker}")
        for p in bad_parents:
            rec(p, depth + 1)

    rec(node, 0)
    return "\n".join(lines)


class LeakageError(Exception):
    """Raised at check time when a decision depends on future information.

    A temporal-non-interference violation: the value used to form the position
    at decision time `time` has availability label A(time) > required.
    """

    def __init__(self, time, inferred, required, source, path=None):
        self.time = time
        self.inferred = inferred
        self.required = required
        self.source = source
        self.path = path
        msg = (f"look-ahead bias: position at t={time} depends on information "
               f"available only at A={_fmt(inferred)} (required A <= {required}); "
               f"offending source: {source}")
        if path:
            msg += "\n  dependency path (future -> decision):\n" + \
                "\n".join("    " + ln for ln in path.splitlines())
        super().__init__(msg)
