"""Representative (illustrative) quant ML snippet with classic CV/scaling leaks."""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

df = pd.read_csv("prices.csv")
df["ret"] = df["close"].pct_change()
X = df[["f1", "f2", "f3"]].fillna(0.0)
y = df["ret"].shift(-1)                      # forward target

# LEAK 1: standardise on the whole sample before splitting
X_scaled = StandardScaler().fit_transform(X)

# LEAK 2: shuffled split on time-series rows
X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y, test_size=0.2)

# LEAK 3: shuffled K-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=0)
model = Ridge().fit(X_tr, y_tr)
