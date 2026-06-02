"""Representative (illustrative) ML snippet written correctly (no leaks)."""
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

df = pd.read_csv("prices.csv")
X = df[["f1", "f2", "f3"]].fillna(0.0)
y = df["close"].pct_change().shift(-1)

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(X):
    # scaler fit inside the fold via a pipeline -> no full-sample leakage
    model = make_pipeline(StandardScaler(), Ridge())
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
