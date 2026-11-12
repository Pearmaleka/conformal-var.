from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from conformal_var import METHODS, calibrate

scores_strategy = st.lists(
    st.floats(-50, 50, allow_nan=False, allow_infinity=False), min_size=40, max_size=400
)


def _errors(scores, alpha, method, **kw):
    """Calibrate a base forecast of 0 against returns -s, so that the scores are s."""
    s = np.asarray(scores, dtype=float)
    out = calibrate(np.zeros(len(s)), -s, alpha, method, **kw)
    done = ~np.isnan(out.correction)
    return (s[done] > out.correction[done]).astype(float), out


@settings(max_examples=200, deadline=None)
@given(
    scores=scores_strategy,
    alpha=st.sampled_from([0.01, 0.05, 0.1, 0.5]),
    gamma=st.floats(0.001, 0.2),
    window=st.integers(1, 30),
)
def test_aci_coverage_bound_holds_for_any_sequence(scores, alpha, gamma, window):
    """Gibbs and Candes (2021), Proposition 4.1."""
    err, _ = _errors(scores, alpha, "aci", gamma=gamma, window=window, cap_at_max=False)
    T = len(err)
    if T:
        bound = (max(alpha, 1 - alpha) + gamma) / (gamma * T)
        assert abs(err.mean() - alpha) <= bound + 1e-12


@settings(max_examples=200, deadline=None)
@given(
    scores=scores_strategy,
    alpha=st.sampled_from([0.01, 0.05, 0.1, 0.5]),
    lr=st.floats(0.01, 5.0),
    window=st.integers(1, 30),
)
def test_quantile_tracking_bound_holds_for_bounded_scores(scores, alpha, lr, window):
    """With scores in [-B, B] and a fixed step, |mean err - alpha| <= (2B + lr) / (lr T)."""
    err, _ = _errors(scores, alpha, "tracking", lr=lr, scale_lr=False, window=window)
    T = len(err)
    if T:
        B = 50.0
        assert abs(err.mean() - alpha) <= (2 * B + lr) / (lr * T) + 1e-12


@pytest.mark.parametrize("method", ["rolling", "aci", "tracking"])
def test_exception_rate_on_iid_scores(method):
    rng = np.random.default_rng(0)
    err, _ = _errors(rng.standard_t(4, 20_000), 0.05, method, window=250)
    assert abs(err.mean() - 0.05) < 0.006


def test_aci_reacts_to_a_variance_break():
    rng = np.random.default_rng(1)
    s = np.r_[rng.standard_normal(3000), 3 * rng.standard_normal(3000)]
    err_aci, _ = _errors(s, 0.01, "aci", window=500)
    err_roll, _ = _errors(s, 0.01, "rolling", window=500)
    after = slice(-2500, None)  # once the window has refilled, both are fine; before, ACI wins
    first = slice(2500, 3000)
    assert err_aci[first].mean() <= err_roll[first].mean()
    assert abs(err_aci.mean() - 0.01) < abs(err_roll.mean() - 0.01) + 0.002
    assert err_aci[after].mean() < 0.02


def test_cap_keeps_forecasts_finite():
    # Ever larger losses: every day is an exception, so alpha_t is driven below zero.
    s = np.r_[np.zeros(100), np.arange(1.0, 21.0), np.zeros(200)]
    _, capped = _errors(s, 0.05, "aci", window=50, gamma=0.05)
    _, raw = _errors(s, 0.05, "aci", window=50, gamma=0.05, cap_at_max=False)
    done = ~np.isnan(capped.q)
    assert np.isfinite(capped.q[done]).all()
    assert np.isinf(raw.q[done]).any()


@pytest.mark.parametrize("method", METHODS)
def test_no_look_ahead(method):
    rng = np.random.default_rng(2)
    base = -1.6 + 0.2 * rng.standard_normal(800)
    r = rng.standard_normal(800)
    a = calibrate(base, r, 0.05, method, window=100)
    r2 = r.copy()
    r2[600:] = rng.standard_normal(200) * 5
    b = calibrate(base, r2, 0.05, method, window=100)
    np.testing.assert_array_equal(a.q[:601], b.q[:601])  # day 600 uses returns up to 599


def test_missing_base_forecasts_are_skipped():
    rng = np.random.default_rng(3)
    base = np.r_[np.full(50, np.nan), np.full(300, -1.6)]
    r = rng.standard_normal(350)
    out = calibrate(base, r, 0.05, "rolling", window=100)
    assert np.isnan(out.q[:150]).all() and np.isfinite(out.q[150:]).all()
    none = calibrate(base, r, 0.05, "none")
    np.testing.assert_array_equal(none.q, base)


def test_validation():
    with pytest.raises(ValueError):
        calibrate([0.0], [0.0], 1.0)
    with pytest.raises(ValueError):
        calibrate([0.0], [0.0], 0.1, "magic")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        calibrate([0.0, 0.0], [0.0], 0.1)
    with pytest.raises(ValueError):
        calibrate([0.0], [np.nan], 0.1)
