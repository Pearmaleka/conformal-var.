"""Base models: one-day-ahead alpha-quantile forecasts of returns.

Every function takes daily returns (in percent) and returns a Series aligned with them, where
the value at day t uses returns up to day t - 1 only. Days without enough history are NaN.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats

__all__ = ["FORECASTERS", "ewma_normal", "garch_t", "gbm_quantile", "historical_simulation"]


def _check(returns: pd.Series, alpha: float) -> None:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if returns.isna().any():
        raise ValueError("returns must not contain NaN")


def historical_simulation(returns: pd.Series, alpha: float, window: int = 250) -> pd.Series:
    """Empirical alpha-quantile of the previous ``window`` returns."""
    _check(returns, alpha)
    return returns.rolling(window).quantile(alpha).shift(1).rename("hs")


def ewma_normal(
    returns: pd.Series, alpha: float, lam: float = 0.94, burn_in: int = 250
) -> pd.Series:
    """RiskMetrics: zero mean, normal innovations and an EWMA variance.

    ``s2_t = lam s2_{t-1} + (1 - lam) r_{t-1}^2``, started from the first squared return.
    """
    _check(returns, alpha)
    var = (returns**2).ewm(alpha=1 - lam, adjust=False).mean().shift(1)
    q = stats.norm.ppf(alpha) * np.sqrt(var)
    q.iloc[:burn_in] = np.nan
    return q.rename("ewma")


def garch_t(
    returns: pd.Series, alpha: float, refit_every: int = 250, min_obs: int = 1000
) -> pd.Series:
    """GARCH(1,1) with Student-t innovations and a constant mean.

    Parameters are estimated by maximum likelihood (``arch``) on all returns before the refit
    date, every ``refit_every`` days. Between refits the variance is updated daily with the
    fixed parameters.
    """
    from arch import arch_model

    _check(returns, alpha)
    r = returns.to_numpy(dtype=float)
    n = len(r)
    q = np.full(n, np.nan)
    for k in range(min_obs, n, refit_every):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = arch_model(r[:k], mean="constant", vol="garch", p=1, q=1, dist="t").fit(
                disp="off"
            )
        p = res.params
        mu, omega, a1, b1, nu = (
            p["mu"],
            p["omega"],
            p["alpha[1]"],
            p["beta[1]"],
            p["nu"],
        )
        end = min(k + refit_every, n)
        # In-sample conditional variance up to k - 1, then one-step updates.
        sigma2_prev = float(res.conditional_volatility[-1] ** 2)
        eps_prev = r[k - 1] - mu
        z = stats.t.ppf(alpha, nu) * np.sqrt((nu - 2) / nu)
        for t in range(k, end):
            sigma2 = omega + a1 * eps_prev**2 + b1 * sigma2_prev
            q[t] = mu + np.sqrt(sigma2) * z
            sigma2_prev, eps_prev = sigma2, r[t] - mu
    return pd.Series(q, index=returns.index, name="garch")


def _features(returns: pd.Series) -> pd.DataFrame:
    """Predictors for the return on day t, built from returns up to t - 1."""
    past = returns.shift(1)
    out = {
        "r1": past,
        "r2": returns.shift(2),
        "abs_r1": past.abs(),
        "min_22": past.rolling(22).min(),
        "ewma_vol": np.sqrt((returns**2).ewm(alpha=0.06, adjust=False).mean().shift(1)),
    }
    for w in (5, 22, 66, 250):
        out[f"vol_{w}"] = past.rolling(w).std()
    return pd.DataFrame(out)


def gbm_quantile(
    returns: pd.Series,
    alpha: float,
    refit_every: int = 250,
    min_obs: int = 1000,
    seed: int = 0,
) -> pd.Series:
    """Gradient-boosted quantile regression on lagged returns and volatility features.

    scikit-learn's ``HistGradientBoostingRegressor`` with the pinball loss, refit every
    ``refit_every`` days on all earlier data.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    _check(returns, alpha)
    X = _features(returns)
    y = returns.to_numpy(dtype=float)
    ok = X.notna().all(axis=1).to_numpy()
    Xv = X.to_numpy()
    n = len(y)
    q = np.full(n, np.nan)
    for k in range(min_obs, n, refit_every):
        train = np.flatnonzero(ok[:k])
        model = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=alpha,
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=8,
            min_samples_leaf=100,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=seed,
        )
        model.fit(Xv[train], y[train])
        end = min(k + refit_every, n)
        rows = np.arange(k, end)
        rows = rows[ok[rows]]
        q[rows] = model.predict(Xv[rows])
    return pd.Series(q, index=returns.index, name="gbm")


FORECASTERS = {
    "hs": historical_simulation,
    "ewma": ewma_normal,
    "garch": garch_t,
    "gbm": gbm_quantile,
}
