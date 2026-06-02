"""Real-market data for the benchmark, with on-disk caching.

The leakage patterns in `patterns.py` take a clean (prices, fwd) panel and are
agnostic to where it came from, so we just need to produce the same shape from
real data. We fetch once, cache to parquet under `data/`, and read the cache on
every subsequent run -- so the benchmark is reproducible and works offline after
the first fetch.

Two sources:
  * equities via yfinance (adjusted closes for a fixed liquid universe),
  * crypto via ccxt / Kraken (daily closes for major pairs).
"""
import hashlib
import os

import numpy as np
import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(CACHE_DIR, exist_ok=True)

# fixed, full-history liquid universe (avoids ad hoc survivorship in the *data*;
# the survivorship *pattern* injects the bias deliberately and explicitly)
EQUITY_UNIVERSE = ["AAPL", "MSFT", "JPM", "XOM", "JNJ", "PG", "KO", "WMT",
                   "GE", "CAT", "MMM", "IBM", "CVX", "MCD", "DIS", "INTC",
                   "CSCO", "PFE", "T", "HD"]
CRYPTO_UNIVERSE = ["BTC/USD", "ETH/USD", "LTC/USD", "XRP/USD", "BCH/USD",
                   "ADA/USD", "LINK/USD", "DOT/USD"]


def _cache_path(key):
    h = hashlib.md5(key.encode()).hexdigest()[:10]
    return os.path.join(CACHE_DIR, f"{key.split('|')[0]}_{h}.pkl")


def _to_arrays(close_df):
    """(dates x assets) closes -> (prices, fwd) arrays with a clean panel.

    Keep only assets with full history over the common date range; forward-fill
    isolated gaps, then drop any remaining incomplete rows.
    """
    close_df = close_df.sort_index().ffill().dropna(axis=1, how="any")
    close_df = close_df.dropna(axis=0, how="any")
    prices = close_df.to_numpy(dtype=float)
    fwd = np.full_like(prices, np.nan)
    fwd[:-1] = prices[1:] / prices[:-1] - 1.0
    return prices, fwd, list(close_df.columns), close_df.index


def load_equities(start="2010-01-01", end="2023-12-31", refresh=False):
    key = f"equities|{start}|{end}|{','.join(EQUITY_UNIVERSE)}"
    path = _cache_path(key)
    if os.path.exists(path) and not refresh:
        close = pd.read_pickle(path)
    else:
        import yfinance as yf
        raw = yf.download(EQUITY_UNIVERSE, start=start, end=end,
                          progress=False, auto_adjust=True)
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        close.to_pickle(path)
    return _to_arrays(close)


def load_crypto(exchange="kraken", limit=720, refresh=False):
    key = f"crypto|{exchange}|{limit}|{','.join(CRYPTO_UNIVERSE)}"
    path = _cache_path(key)
    if os.path.exists(path) and not refresh:
        close = pd.read_pickle(path)
    else:
        import ccxt
        ex = getattr(ccxt, exchange)({"timeout": 20000, "enableRateLimit": True})
        cols = {}
        for sym in CRYPTO_UNIVERSE:
            try:
                ohlcv = ex.fetch_ohlcv(sym, timeframe="1d", limit=limit)
            except Exception:
                continue
            s = pd.Series({pd.Timestamp(t, unit="ms"): c for t, _, _, _, c, _ in ohlcv})
            cols[sym] = s
        close = pd.DataFrame(cols)
        close.to_pickle(path)
    return _to_arrays(close)
