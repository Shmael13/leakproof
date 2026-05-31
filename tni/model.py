"""Model fitting under availability labels (DESIGN.md A.4, model rule).

A fitted model carries the label `tau = join of the availabilities of every
training datum it saw` (features AND targets). A prediction at row s then has
availability `max(tau_of_the_model_used, A_features(s))`. Using that prediction
to trade at s is legal only if it is <= s.

This is exactly what separates cross-validation schemes:
  * walk-forward / expanding window: the model for row s is trained only on
    rows whose data is available by s, so tau <= s -> SAFE.
  * shuffled K-fold: a fold's model is trained on other folds that include
    *future* rows, so tau reaches the end of the sample -> predictions at
    interior rows have availability > s -> LEAK.

We return predictions as a TemporalSeries, so the ordinary legality check in
`to_positions` catches improper CV with no special-casing.
"""
from typing import Protocol, cast

import numpy as np

from .core import TOP
from .series import TemporalSeries


class Estimator(Protocol):
    """A scikit-learn-style estimator: the bit of the API we actually use."""
    def fit(self, X, y) -> "Estimator": ...
    def predict(self, X): ...


def _fit_predict(estimator: Estimator, X, y, train_idx, test_idx):
    from sklearn.base import clone
    # clone() is typed to also accept lists/tuples/sets of estimators, so its
    # inferred return type is a union; pin it back to a single estimator.
    m = cast(Estimator, clone(estimator))
    m.fit(X[train_idx], y[train_idx])
    return m.predict(X[test_idx])


def predict_walkforward(estimator: Estimator, X, y, xavail, yavail,
                        min_train=200, retrain_every=20, embargo=0, label="wf"):
    """Expanding-window out-of-sample predictions; SAFE by construction.

    For row s the model is trained on rows [0, s - embargo) only, so its label
    is max avail over those rows (<= s for clean, lagged features and a target
    realised by s). The `embargo` gap is the purged/embargoed-CV discipline: it
    drops the rows whose target window would overlap the test point.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    T = len(X)
    preds = np.full(T, np.nan)
    avail = np.full(T, TOP)

    s = min_train
    while s < T:
        block_end = min(T, s + retrain_every)
        train = np.arange(0, max(0, s - embargo))
        train = train[np.isfinite(y[train])]
        if len(train) >= 2:
            test = np.arange(s, block_end)
            preds[test] = _fit_predict(estimator, X, y, train, test)
            tau = max(xavail[train].max(), yavail[train].max())
            for t in test:
                avail[t] = max(tau, xavail[t])
        s = block_end
    return TemporalSeries(preds, avail, label=f"{label}_pred")


def predict_kfold(estimator: Estimator, X, y, xavail, yavail, n_splits=5,
                  shuffle=True, seed=0, label="kfold"):
    """K-fold cross-val predictions. Shuffled K-fold on time series LEAKS:
    each fold's model sees future rows, so its label reaches the sample end."""
    from sklearn.model_selection import KFold
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    T = len(X)
    preds = np.full(T, np.nan)
    avail = np.full(T, TOP)

    valid = np.where(np.isfinite(y))[0]
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed if shuffle else None)
    for train_rel, test_rel in kf.split(valid):
        train, test = valid[train_rel], valid[test_rel]
        preds[test] = _fit_predict(estimator, X, y, train, test)
        tau = max(xavail[train].max(), yavail[train].max())   # includes future
        for t in test:
            avail[t] = max(tau, xavail[t])
    return TemporalSeries(preds, avail, label=f"{label}_pred")


def make_lagged_features(prices, lags=(1, 2, 3, 5, 10)):
    """Build a lagged-return feature matrix X (T, F) and forward-return target.

    Feature X[t, j] is the return realised `lags[j]` days before t, so every
    feature in row t is available by t (xavail[t] = t). The target fwd[t] is the
    return over (t -> t+1), available at t+1 (yavail[t] = t+1).
    """
    prices = np.asarray(prices, float)
    T = len(prices)
    ret = np.full(T, np.nan)
    ret[1:] = prices[1:] / prices[:-1] - 1.0

    X = np.zeros((T, len(lags)))
    for j, k in enumerate(lags):
        X[k:, j] = ret[:T - k]          # return from k days earlier
    X = np.nan_to_num(X, nan=0.0)

    fwd = np.full(T, np.nan)
    fwd[:-1] = prices[1:] / prices[:-1] - 1.0
    xavail = np.arange(T, dtype=float)
    yavail = np.arange(T, dtype=float) + 1.0
    return X, fwd, xavail, yavail
