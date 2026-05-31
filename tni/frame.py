"""TemporalFrame: a panel of time-indexed values, shape (T, A) = (time, assets).

Each cell carries an availability label, exactly like TemporalSeries. Temporal
operators (shift, rolling, pct_change) act down the time axis, per asset.
Cross-sectional operators act across the asset axis at a *fixed* time s: their
output availability at row s is the join over that row only, so for a clean
panel A(s, .) == s -- SAFE. Panel-pooled operators touch every (time, asset)
cell at once, so their label is the global join (end-of-sample / TOP) -- LEAK
when used at any interior decision time.

Like TemporalSeries, every node keeps its operands (`_parents`) and operator
name (`_op`) so a LeakageError can print the dependency path.
"""
import numpy as np

from .core import TOP, LeakageError, explain_leak


class TemporalFrame:
    def __init__(self, values, avail, label="<frame>", parents=(), op=None):
        self.values = np.asarray(values, dtype=float)   # (T, A)
        self.avail = np.asarray(avail, dtype=float)
        self.label = label
        self._parents = tuple(parents)
        self._op = op or label
        assert self.values.shape == self.avail.shape
        assert self.values.ndim == 2

    @property
    def shape(self):
        return self.values.shape

    @classmethod
    def observe(cls, values, avail_lag=0, label="obs"):
        """Panel where row s (across all assets) is knowable at s + avail_lag."""
        values = np.asarray(values, dtype=float)
        T = values.shape[0]
        avail = (np.arange(T, dtype=float) + float(avail_lag))[:, None]
        avail = np.broadcast_to(avail, values.shape).copy()
        name = f"{label}(lag={avail_lag})"
        return cls(values, avail, label=name, op=name)

    # --- temporal operators (down the time axis, per asset) ---
    def shift(self, k, label=None):
        v = np.full_like(self.values, np.nan)
        a = np.full_like(self.avail, TOP)
        T = self.shape[0]
        for s in range(T):
            j = s - k
            if 0 <= j < T:
                v[s] = self.values[j]
                a[s] = self.avail[j]
        return TemporalFrame(v, a, label=label or f"shift({self.label},{k})",
                             parents=(self,), op=f"shift(k={k})")

    def pct_change(self, label=None):
        v = np.full_like(self.values, np.nan)
        a = np.full_like(self.avail, TOP)
        with np.errstate(divide="ignore", invalid="ignore"):
            v[1:] = np.where(self.values[:-1] != 0,
                             self.values[1:] / self.values[:-1] - 1.0, np.nan)
        a[1:] = np.maximum(self.avail[1:], self.avail[:-1])
        return TemporalFrame(v, a, label=label or f"pct_change({self.label})",
                             parents=(self,), op="pct_change")

    def rolling_back(self, window, reduce="mean", label=None):
        fns = {"mean": np.mean, "std": lambda a: np.std(a, ddof=1),
               "sum": np.sum, "max": np.max, "min": np.min}
        fn = fns[reduce]
        T, A = self.shape
        v = np.full_like(self.values, np.nan)
        a = np.full_like(self.avail, TOP)
        for s in range(T):
            lo = max(0, s - window + 1)
            block = self.values[lo:s + 1]
            ablock = self.avail[lo:s + 1]
            mask = np.isfinite(block)
            for j in range(A):
                col = block[:, j][mask[:, j]]
                if len(col) and (reduce != "std" or len(col) > 1):
                    with np.errstate(invalid="ignore", divide="ignore"):
                        v[s, j] = fn(col)
                # join only over finite (actually-contributing) rows
                if mask[:, j].any():
                    a[s, j] = ablock[:, j][mask[:, j]].max()
        return TemporalFrame(v, a, parents=(self,), op=f"roll_back{window}.{reduce}",
                             label=label or f"roll_back{window}.{reduce}({self.label})")

    # --- elementwise binops: label = join ---
    def _binop(self, other, fn, sym):
        a = np.maximum(self.avail, other.avail)
        return TemporalFrame(fn(self.values, other.values), a,
                             label=f"({self.label}{sym}{other.label})",
                             parents=(self, other), op=f"binop({sym})")

    def __sub__(self, o): return self._binop(o, lambda x, y: x - y, "-")
    def __add__(self, o): return self._binop(o, lambda x, y: x + y, "+")
    def __mul__(self, o): return self._binop(o, lambda x, y: x * y, "*")

    def scale(self, c, label=None):
        """Multiply by a constant (availability unchanged)."""
        return TemporalFrame(self.values * c, self.avail.copy(),
                             label=label or f"({c}*{self.label})",
                             parents=(self,), op=f"scale({c})")

    def __truediv__(self, o):
        def d(x, y):
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(y != 0, x / y, np.nan)
        return self._binop(o, d, "/")

    # --- cross-sectional ops (across assets at fixed time s): SAFE ---
    def _xs(self, transform, name):
        v = np.full_like(self.values, np.nan)
        a = np.full_like(self.avail, TOP)
        for s in range(self.shape[0]):
            row = self.values[s]
            mask = np.isfinite(row)
            if mask.sum() >= 1:
                v[s, mask] = transform(row[mask])
                # a cross-sectional value depends on the other assets *present*
                # at time s; NaN assets do not participate, so the join is over
                # the finite cells only (-> stays at s for a clean panel).
                a[s] = self.avail[s][mask].max()
        return TemporalFrame(v, a, label=f"{name}_xs({self.label})",
                             parents=(self,), op=f"{name}_xs")

    def rank_xs(self):
        def r(x):
            order = np.argsort(np.argsort(x))
            return order / max(len(x) - 1, 1)        # in [0, 1]
        return self._xs(r, "rank")

    def zscore_xs(self):
        def z(x):
            sd = x.std(ddof=1) if len(x) > 1 else np.nan
            return (x - x.mean()) / sd if sd else np.zeros_like(x)
        return self._xs(z, "zscore")

    def demean_xs(self):
        return self._xs(lambda x: x - x.mean(), "demean")

    # --- panel-pooled op (touches the whole panel at once): LEAK ---
    def zscore_panel(self):
        finite = self.values[np.isfinite(self.values)]
        mu = finite.mean() if len(finite) else np.nan
        sd = finite.std(ddof=1) if len(finite) > 1 else np.nan
        with np.errstate(divide="ignore", invalid="ignore"):
            v = (self.values - mu) / sd
        a = np.full_like(self.avail, TOP)             # depends on all of time
        return TemporalFrame(v, a, label=f"zscore_panel({self.label})",
                             parents=(self,), op="zscore_panel")

    # --- legality check ---
    def to_positions(self, exec_lag=0):
        T, A = self.shape
        for s in range(T):
            required = s - exec_lag
            for j in range(A):
                if not np.isfinite(self.values[s, j]):
                    continue
                if self.avail[s, j] > required:
                    raise LeakageError(
                        time=s, inferred=self.avail[s, j], required=required,
                        source=f"{self.label}[asset {j}]",
                        path=explain_leak(self, s, required),
                    )
        return self
