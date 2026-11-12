from __future__ import annotations

import hashlib
from importlib import resources

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from conformal_var import (
    ASSETS,
    FORECASTERS,
    ewma_normal,
    garch_t,
    historical_simulation,
    load_returns,
)


def simulate_garch_t(n: int, seed: int, omega=0.05, a1=0.08, b1=0.9, nu=6.0, mu=0.03):
    rng = np.random.default_rng(seed)
    z = stats.t.rvs(nu, size=n, random_state=rng) * np.sqrt((nu - 2) / nu)
    r = np.empty(n)
    s2 = np.empty(n)
    s2[0] = omega / (1 - a1 - b1)
    r[0] = mu + np.sqrt(s2[0]) * z[0]
    for t in range(1, n):
        s2[t] = omega + a1 * (r[t - 1] - mu) ** 2 + b1 * s2[t - 1]
        r[t] = mu + np.sqrt(s2[t]) * z[t]
    q_true = mu + np.sqrt(s2) * stats.t.ppf(0.05, nu) * np.sqrt((nu - 2) / nu)
    idx = pd.bdate_range("2000-01-03", periods=n)
    return pd.Series(r, index=idx), q_true


@pytest.fixture(scope="module")
def garch_series():
    return simulate_garch_t(5000, seed=0)


def test_garch_recovers_the_true_quantile(garch_series):
    r, q_true = garch_series
    q = garch_t(r, 0.05).to_numpy()
    ok = ~np.isnan(q)
    assert np.corrcoef(q[ok], q_true[ok])[0, 1] > 0.97
    assert abs(np.mean(r.to_numpy()[ok] < q[ok]) - 0.05) < 0.012


@pytest.mark.parametrize("name", sorted(FORECASTERS))
def test_forecasters_do_not_look_ahead(name, garch_series):
    r = garch_series[0].iloc[:1500]
    kw = {"refit_every": 200, "min_obs": 1000} if name in ("garch", "gbm") else {}
    a = FORECASTERS[name](r, 0.05, **kw)
    r2 = r.copy()
    r2.iloc[1300:] = r2.iloc[1300:] * 10
    b = FORECASTERS[name](r2, 0.05, **kw)
    pd.testing.assert_series_equal(a.iloc[:1301], b.iloc[:1301])
    assert a.iloc[1350:].notna().all()


def test_historical_simulation_and_ewma_by_hand(garch_series):
    r = garch_series[0].iloc[:600]
    hs = historical_simulation(r, 0.05, window=250)
    assert hs.iloc[300] == pytest.approx(np.quantile(r.iloc[50:300], 0.05))
    assert np.isnan(hs.iloc[249])
    ew = ewma_normal(r, 0.05, lam=0.94, burn_in=10)
    s2 = r.iloc[0] ** 2
    for t in range(1, 400):
        s2 = 0.94 * s2 + 0.06 * r.iloc[t - 1] ** 2 if t > 1 else s2
        if t >= 10:
            assert ew.iloc[t] == pytest.approx(stats.norm.ppf(0.05) * np.sqrt(s2))


def test_inputs_are_validated():
    r = pd.Series([0.1, np.nan, 0.2])
    with pytest.raises(ValueError):
        historical_simulation(r, 0.05)
    with pytest.raises(ValueError):
        historical_simulation(pd.Series([0.1]), 0.0)


@pytest.mark.parametrize("name", sorted(ASSETS))
def test_data(name):
    r = load_returns(name)
    assert len(r) > 4000
    assert r.notna().all() and r.index.is_monotonic_increasing and r.index.is_unique
    assert 0.3 < r.std() < 3.0


def test_vendored_file_is_the_pinned_copy():
    path = resources.files("conformal_var") / "_data" / "usdjpy_daily.csv"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == "aadc16e3a55c3ebdb1f3e0aed8d7b5800d013eaa8ead2ac314143a56814408c3"


def test_unknown_asset():
    with pytest.raises(ValueError):
        load_returns("bitcoin")
