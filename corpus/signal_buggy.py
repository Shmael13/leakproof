"""Representative (illustrative) signal-construction snippet with timing leaks."""
import pandas as pd

df = pd.read_csv("prices.csv")
ret = df["close"].pct_change()

# LEAK 1: centered smoothing uses future bars
smooth = ret.rolling(11, center=True).mean()

# LEAK 2: full-sample standardisation
zscore = (smooth - smooth.mean()) / smooth.std()

# LEAK 3: aligning the signal with a forward shift (lead)
signal = zscore.shift(-1)
df["position"] = signal.apply(lambda x: 1 if x > 0 else -1)
