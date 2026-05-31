"""TemporalSeries: a time-indexed value carrying a per-row availability label.

For a row at logical time s, `avail[s]` is the time at which that row's value
first becomes knowable. A clean backward-looking series has avail[s] == s.
Each operator below propagates labels via its availability transfer function
(DESIGN.md A.4). The legality check lives in `to_positions`.

Evaluation is eager (values + labels computed at build time), but every node
also keeps a link to the operands it was built from (`_parents`) and the name
of the operator (`_op`). That retained provenance is what lets a LeakageError
print the *dependency path* from the future datum down to the decision, even
though we never build a separate lazy graph. The leak is still caught in
`to_positions` -- before any PnL/backtest runs (build -> check -> run).
"""
import numpy as np

from .core import BOT, TOP, LeakageError, explain_leak


class TemporalSeries:
    def __init__(self, values, avail, label="<expr>", parents=(), op=None):
        self.values = np.asarray(values, dtype=float)
        self.avail = np.asarray(avail, dtype=float)
        self.label = label
        self._parents = tuple(parents)
        self._op = op or label
        assert self.values.shape == self.avail.shape

    def __len__(self):
        return len(self.values)

    @classmethod
    def constant(cls, value, n, label=None):
        """A literal broadcast across n rows; available for all time (BOT)."""
        return cls(np.full(n, float(value)), np.full(n, BOT),
                   label=label or f"const({value})", op=f"const({value})")

    @classmethod
    def observe(cls, values, avail_lag=0, label="obs"):
        """A raw observation whose row s is knowable at time s + avail_lag.

        A daily CLOSE is known only at the end of its bar -> avail_lag=1
        (available at the next decision point); an OPEN -> avail_lag=0.
        """
        values = np.asarray(values, dtype=float)
        n = len(values)
        avail = np.arange(n, dtype=float) + float(avail_lag)
        name = f"{label}(lag={avail_lag})"
        return cls(values, avail, label=name, op=name)

    # --- temporal operators (availability transfer functions) ---
    def shift(self, k, label=None):
        """Shift by k. k>0 is a backward lag (safe); k<0 is a forward lead (leaks)."""
        n = len(self)
        v = np.full(n, np.nan)
        a = np.full(n, TOP)
        for s in range(n):
            j = s - k
            if 0 <= j < n:
                v[s] = self.values[j]
                a[s] = self.avail[j]
        return TemporalSeries(v, a, label=label or f"shift({self.label},{k})",
                              parents=(self,), op=f"shift(k={k})")

    def pct_change(self, label=None):
        n = len(self)
        v = np.full(n, np.nan)
        a = np.full(n, TOP)
        for s in range(1, n):
            prev = self.values[s - 1]
            if prev != 0:
                v[s] = self.values[s] / prev - 1.0
            a[s] = max(self.avail[s], self.avail[s - 1])  # join of operands
        return TemporalSeries(v, a, label=label or f"pct_change({self.label})",
                              parents=(self,), op="pct_change")

    def sign(self, label=None):
        return TemporalSeries(
            np.sign(self.values), self.avail.copy(),
            label=label or f"sign({self.label})", parents=(self,), op="sign",
        )

    def scale(self, c, label=None):
        """Multiply by a constant (availability unchanged)."""
        return TemporalSeries(self.values * c, self.avail.copy(),
                              label=label or f"({c}*{self.label})",
                              parents=(self,), op=f"scale({c})")

    def _binop(self, other, fn, sym):
        a = np.maximum(self.avail, other.avail)  # label = join of operands
        return TemporalSeries(
            fn(self.values, other.values), a,
            label=f"({self.label}{sym}{other.label})",
            parents=(self, other), op=f"binop({sym})",
        )

    def __sub__(self, other):
        return self._binop(other, lambda x, y: x - y, "-")

    def __add__(self, other):
        return self._binop(other, lambda x, y: x + y, "+")

    def __mul__(self, other):
        return self._binop(other, lambda x, y: x * y, "*")

    def __truediv__(self, other):
        def safe_div(x, y):
            with np.errstate(divide="ignore", invalid="ignore"):
                out = np.where(y != 0, x / y, np.nan)
            return out
        return self._binop(other, safe_div, "/")

    # --- rolling windows (availability = join over the window) ---
    def rolling(self, window, mode="back"):
        """Return a Rolling handle. mode in {back, center, forward}.

        back:    window [s-w+1 .. s]      -> A(s) = s for a clean series (SAFE)
        center:  window [s-w//2 .. s+w//2] -> includes future          (LEAK)
        forward: window [s .. s+w-1]        -> includes future          (LEAK)
        """
        return Rolling(self, window, mode)

    def expanding(self):
        """Expanding window [0 .. s]. Backward-looking, so SAFE for a clean series."""
        return Rolling(self, window=None, mode="expanding")

    # --- full-sample reductions (touch ALL time -> label TOP -> LEAK on use) ---
    def _full_reduce(self, fn, name):
        finite = self.values[np.isfinite(self.values)]
        with np.errstate(invalid="ignore", divide="ignore"):
            val = fn(finite) if len(finite) else np.nan
        n = len(self)
        # the reduced scalar depends on every row, so its availability is the
        # join over the whole series; broadcasting it back gives label TOP.
        return TemporalSeries(np.full(n, val), np.full(n, TOP),
                              label=f"full_{name}({self.label})",
                              parents=(self,), op=f"full_{name}")

    def full_mean(self):
        return self._full_reduce(np.mean, "mean")

    def full_std(self):
        return self._full_reduce(lambda a: np.std(a, ddof=1), "std")

    def full_min(self):
        return self._full_reduce(np.min, "min")

    def full_max(self):
        return self._full_reduce(np.max, "max")

    def full_quantile(self, q):
        return self._full_reduce(lambda a: np.quantile(a, q), f"q{q}")

    # --- resample to a lower frequency, labelled at period end ---
    def resample(self, period, reduce="mean", lag_periods=0, label=None):
        """Aggregate rows into blocks of `period` and broadcast back.

        A period aggregate is only knowable at the period END (its label is the
        join over the period's rows). Consuming period p's aggregate during
        period p (lag_periods=0) therefore leaks at every intraperiod row; the
        correct usage lags by one period (lag_periods=1).
        """
        fns = {"mean": np.mean, "sum": np.sum, "max": np.max,
               "min": np.min, "last": lambda a: a[-1]}
        fn = fns[reduce]
        n = len(self)
        v = np.full(n, np.nan)
        a = np.full(n, TOP)
        n_periods = (n + period - 1) // period
        for p in range(n_periods):
            lo, hi = p * period, min(n, (p + 1) * period)
            vals, av = self.values[lo:hi], self.avail[lo:hi]
            fmask = np.isfinite(vals)
            if not fmask.any():
                continue
            with np.errstate(invalid="ignore"):
                pv = fn(vals[fmask])
            pe = av[fmask].max()                      # available at period end
            cp = p + lag_periods                      # consuming period
            clo, chi = cp * period, min(n, (cp + 1) * period)
            if clo < n:
                v[clo:chi] = pv
                a[clo:chi] = pe
        return TemporalSeries(v, a, parents=(self,),
                              op=f"resample{period}.{reduce}(lag={lag_periods})",
                              label=label or f"resample{period}.{reduce}({self.label})")

    # --- convenience composites ---
    def zscore_full(self):
        """Full-sample standardisation: the classic look-ahead bug (LEAK)."""
        return (self - self.full_mean()) / self.full_std()

    def zscore_back(self, window):
        """Backward-rolling standardisation: the correct version (SAFE)."""
        r = self.rolling(window, "back")
        return (self - r.mean()) / r.std()

    def clip(self, lower, upper, label=None):
        """Clip values to [lower, upper]; bounds are TemporalSeries (their
        availability flows in via the join), so full-sample bounds -> LEAK."""
        lo = lower if isinstance(lower, TemporalSeries) else \
            TemporalSeries.constant(lower, len(self))
        hi = upper if isinstance(upper, TemporalSeries) else \
            TemporalSeries.constant(upper, len(self))
        a = np.maximum(self.avail, np.maximum(lo.avail, hi.avail))
        v = np.clip(self.values, lo.values, hi.values)
        return TemporalSeries(v, a, label=label or f"clip({self.label})",
                              parents=(self, lo, hi), op="clip")

    # --- the legality check: build -> CHECK -> run ---
    def to_positions(self, exec_lag=0):
        """Validate temporal non-interference, then return self as positions.

        Requires A(s) <= s - exec_lag at every decision time s where a position
        is taken (finite value). Raises LeakageError on the first violation.
        """
        n = len(self)
        for s in range(n):
            if not np.isfinite(self.values[s]):
                continue  # no position taken at this time
            required = s - exec_lag
            if self.avail[s] > required:
                raise LeakageError(
                    time=s, inferred=self.avail[s], required=required,
                    source=self.label, path=explain_leak(self, s, required),
                )
        return self


class Rolling:
    """A windowed view over a TemporalSeries.

    The output availability at s is the JOIN (max) of input availabilities over
    the window's index set. Whether that is <= s is decided entirely by which
    indices the window covers -- which is exactly what distinguishes a safe
    backward window from a leaking centered/forward window. No special-casing:
    the leak falls out of the transfer function.
    """

    _MODES = {"back", "center", "forward", "expanding"}

    def __init__(self, series, window, mode):
        if mode not in self._MODES:
            raise ValueError(f"unknown rolling mode: {mode}")
        self.series = series
        self.window = window
        self.mode = mode

    def _index_set(self, s, n):
        w = self.window
        if self.mode == "expanding":
            return range(0, s + 1)
        if self.mode == "back":
            return range(max(0, s - w + 1), s + 1)
        if self.mode == "forward":
            return range(s, min(n, s + w))
        # center
        half = w // 2
        return range(max(0, s - half), min(n, s + half + 1))

    def _agg(self, reduce_fn, name):
        src = self.series
        n = len(src)
        v = np.full(n, np.nan)
        a = np.full(n, TOP)
        for s in range(n):
            idx = np.array(list(self._index_set(s, n)))
            vals = src.values[idx]
            fmask = np.isfinite(vals)
            if fmask.any():
                with np.errstate(invalid="ignore", divide="ignore"):
                    v[s] = reduce_fn(vals[fmask])
                # join only over the data that actually influences the output;
                # NaN inputs are filtered out, so they do not flow.
                a[s] = np.max(src.avail[idx][fmask])
        wlabel = "exp" if self.mode == "expanding" else f"{self.mode}{self.window}"
        return TemporalSeries(v, a, label=f"roll_{wlabel}.{name}({src.label})",
                              parents=(src,), op=f"roll_{wlabel}.{name}")

    def mean(self):
        return self._agg(np.mean, "mean")

    def std(self):
        return self._agg(lambda a: np.std(a, ddof=1) if len(a) > 1 else np.nan, "std")

    def min(self):
        return self._agg(np.min, "min")

    def max(self):
        return self._agg(np.max, "max")

    def sum(self):
        return self._agg(np.sum, "sum")

    def quantile(self, q):
        return self._agg(lambda a: np.quantile(a, q), f"q{q}")
