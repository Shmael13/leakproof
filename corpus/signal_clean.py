"""Representative (illustrative) signal snippet written correctly (no leaks)."""
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

df = pd.read_csv("prices.csv")
ret = df["close"].pct_change()

# trailing window only, and a backward shift to trade on yesterday's info
smooth = ret.rolling(11).mean()
zscore = (smooth - smooth.rolling(60).mean()) / smooth.rolling(60).std()
signal = zscore.shift(1)
df["position"] = signal.apply(lambda x: 1 if x > 0 else -1)

# time-ordered cross-validation
tscv = TimeSeriesSplit(n_splits=5)
