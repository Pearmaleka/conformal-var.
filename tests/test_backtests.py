from __future__ import annotations

import math

import numpy as np
import pytest

from conformal_var import (
    backtest,
    basel_zones,
    christoffersen,
    dynamic_quantile,
    kupiec,
    pinball_loss,
)


def test_kupiec_matches_the_formula():
    hits = np.r_[np.ones(5), np.zeros(245)]
    lr, p = kupiec(hits, 0.01)
    null = 245 * math.log(0.99) + 5 * math.log(0.01)
    alt = 245 * math.log(0.98) + 5 * math.log(0.02)
    assert lr == pytest.approx(-2 * (null - alt))
    assert 0.1 < p < 0.2  # 5 exceptions in 250 days at 99% is not a rejection


def test_kupiec_edge_cases():
    lr, p = kupiec(np.zeros(500), 0.01)
    assert lr == pytest.approx(-2 * 500 * math.log(0.99))
    assert kupiec(np.zeros(10), 0.01)[1] > 0.5
    lr, p = kupiec(np.ones(20), 0.01)
    assert p < 1e-10


def _size(test, alpha: float, n: int, reps: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    pvals = [test(rng.random(n) < alpha) for _ in range(reps)]
    return float(np.mean(np.array(pvals) < 0.05))


def test_tests_hold_their_size_under_the_null():
    alpha = 0.05
    assert 0.02 <= _size(lambda h: kupiec(h, alpha)[1], alpha, 750, 1500, 1) <= 0.08
    assert 0.02 <= _size(lambda h: christoffersen(h, alpha)["p_cc"], alpha, 750, 1500, 2) <= 0.08
    rng = np.random.default_rng(3)
    q = -1.65 + 0.1 * rng.standard_normal(750)
    assert 0.02 <= _size(lambda h: dynamic_quantile(h, q, alpha)[1], alpha, 750, 1500, 4) <= 0.09


def _clustered(n: int, p01: float, p11: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h = np.zeros(n, dtype=int)
    for t in range(1, n):
        h[t] = rng.random() < (p11 if h[t - 1] else p01)
    return h


def test_clustered_exceptions_are_detected():
    # Stationary exception rate p01 / (1 - p11 + p01) = 0.05, but in clusters.
    h = _clustered(2000, p01=0.0263, p11=0.5, seed=0)
    assert abs(h.mean() - 0.05) < 0.02
    assert christoffersen(h, 0.05)["p_ind"] < 0.001
    assert dynamic_quantile(h, np.full(len(h), -1.65), 0.05)[1] < 0.001


def test_christoffersen_counts():
    h = np.array([0, 0, 1, 1, 0, 1, 0, 0, 0, 1])
    # transitions: 00:3 01:3 10:2 11:1
    out = christoffersen(h, 0.3)
    pi01, pi11, pi = 3 / 6, 1 / 3, 4 / 9
    null = 5 * math.log(1 - pi) + 4 * math.log(pi)
    alt = 3 * math.log(1 - pi01) + 3 * math.log(pi01) + 2 * math.log(1 - pi11) + math.log(pi11)
    assert out["lr_ind"] == pytest.approx(-2 * (null - alt))
    assert out["lr_cc"] == pytest.approx(out["lr_ind"] + kupiec(h, 0.3)[0])


def test_pinball_loss():
    assert pinball_loss([1.0, -1.0], [0.0, 0.0], 0.1) == pytest.approx(0.5)
    # The true quantile minimises the expected loss.
    rng = np.random.default_rng(0)
    r = rng.standard_normal(200_000)
    losses = {c: pinball_loss(r, np.full_like(r, c), 0.05) for c in (-2.0, -1.645, -1.3)}
    assert min(losses, key=losses.get) == -1.645


def test_basel_zones():
    h = np.zeros(400, dtype=int)
    h[100:110] = 1
    z = basel_zones(h)  # 151 windows, starting on days 0 to 150
    assert z["max_exceptions"] == 10
    assert z["red"] == pytest.approx(101 / 151)  # starts 0-100 contain all ten
    assert z["yellow"] == pytest.approx(5 / 151)  # starts 101-105 contain 9 down to 5
    assert z["green"] == pytest.approx(45 / 151)


def test_backtest_summary_and_validation():
    rng = np.random.default_rng(1)
    r = rng.standard_normal(2000)
    res = backtest(r, np.full(2000, -2.326), 0.01)
    assert res.n == 2000 and res.p_kupiec > 0.01
    assert res.mean_var == pytest.approx(2.326)
    assert set(res.as_dict()) >= {"hit_rate", "p_dq", "pinball"}
    with pytest.raises(ValueError):
        backtest(r, np.full(1999, -2.0), 0.01)
    with pytest.raises(ValueError):
        backtest(np.r_[r[:-1], np.nan], np.full(2000, -2.0), 0.01)
    with pytest.raises(ValueError):
        kupiec([0, 2], 0.01)
    with pytest.raises(ValueError):
        kupiec([0, 1], 1.5)
