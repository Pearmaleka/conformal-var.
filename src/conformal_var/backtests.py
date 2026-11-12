"""Standard Value-at-Risk backtests.

Conventions: ``q`` is the forecast alpha-quantile of the return (a negative number for a long
position), ``hits`` are the exceptions ``returns < q``.

References
----------
Kupiec, P. (1995). Techniques for verifying the accuracy of risk measurement models.
    Journal of Derivatives, 3(2).
Christoffersen, P. (1998). Evaluating interval forecasts. International Economic Review, 39(4).
Engle, R. and Manganelli, S. (2004). CAViaR: conditional autoregressive value at risk by
    regression quantiles. Journal of Business & Economic Statistics, 22(4).
Basel Committee on Banking Supervision (1996). Supervisory framework for the use of
    "backtesting" in conjunction with the internal models approach to market risk capital.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

__all__ = [
    "BacktestResult",
    "backtest",
    "basel_zones",
    "christoffersen",
    "dynamic_quantile",
    "kupiec",
    "pinball_loss",
]


def _xlogy(x: float, y: float) -> float:
    """x * log(y) with the convention 0 * log(0) = 0."""
    return 0.0 if x == 0 else float(x * np.log(y))


def _as_hits(hits: ArrayLike) -> np.ndarray:
    h = np.asarray(hits)
    if h.ndim != 1 or len(h) == 0:
        raise ValueError("hits must be a non-empty 1-d array")
    if not np.isin(h, (0, 1)).all():
        raise ValueError("hits must contain only 0 and 1")
    return h.astype(int)


def _check_alpha(alpha: float) -> None:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")


def kupiec(hits: ArrayLike, alpha: float) -> tuple[float, float]:
    """Proportion-of-failures likelihood-ratio test. Returns (LR statistic, p-value), chi2(1)."""
    _check_alpha(alpha)
    h = _as_hits(hits)
    n, x = len(h), int(h.sum())
    pi = x / n
    loglik_null = _xlogy(n - x, 1 - alpha) + _xlogy(x, alpha)
    loglik_alt = _xlogy(n - x, 1 - pi) + _xlogy(x, pi)
    lr = max(-2.0 * (loglik_null - loglik_alt), 0.0)
    return lr, float(stats.chi2.sf(lr, 1))


def christoffersen(hits: ArrayLike, alpha: float) -> dict[str, float]:
    """Independence and conditional-coverage tests (first-order Markov alternative).

    Returns LR statistics and p-values: ``lr_ind``/``p_ind`` (chi2(1)) and
    ``lr_cc``/``p_cc`` (Kupiec + independence, chi2(2)).
    """
    h = _as_hits(hits)
    prev, cur = h[:-1], h[1:]
    n00 = int(np.sum((prev == 0) & (cur == 0)))
    n01 = int(np.sum((prev == 0) & (cur == 1)))
    n10 = int(np.sum((prev == 1) & (cur == 0)))
    n11 = int(np.sum((prev == 1) & (cur == 1)))
    pi01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    pi11 = n11 / (n10 + n11) if n10 + n11 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    loglik_null = _xlogy(n00 + n10, 1 - pi) + _xlogy(n01 + n11, pi)
    loglik_alt = (
        _xlogy(n00, 1 - pi01) + _xlogy(n01, pi01) + _xlogy(n10, 1 - pi11) + _xlogy(n11, pi11)
    )
    lr_ind = max(-2.0 * (loglik_null - loglik_alt), 0.0)
    lr_uc, _ = kupiec(h, alpha)
    lr_cc = lr_uc + lr_ind
    return {
        "lr_ind": lr_ind,
        "p_ind": float(stats.chi2.sf(lr_ind, 1)),
        "lr_cc": lr_cc,
        "p_cc": float(stats.chi2.sf(lr_cc, 2)),
    }


def dynamic_quantile(
    hits: ArrayLike, q: ArrayLike, alpha: float, lags: int = 4
) -> tuple[float, float]:
    """Engle and Manganelli's dynamic quantile test.

    Regresses ``hit_t - alpha`` on a constant, ``lags`` lagged values of it and the VaR
    forecast. Under correct conditional coverage all coefficients are zero and
    ``DQ = b' X'X b / (alpha (1 - alpha))`` is chi2 with ``lags + 2`` degrees of freedom.
    Returns (DQ statistic, p-value).
    """
    _check_alpha(alpha)
    h = _as_hits(hits).astype(float) - alpha
    qv = np.asarray(q, dtype=float)
    if qv.shape != h.shape:
        raise ValueError("hits and q must have the same length")
    if not np.isfinite(qv).all():
        raise ValueError("q must be finite")
    n = len(h)
    if n <= 2 * (lags + 2):
        raise ValueError("too few observations for the requested number of lags")
    y = h[lags:]
    cols = [np.ones(n - lags)] + [h[lags - k : n - k] for k in range(1, lags + 1)] + [qv[lags:]]
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    dq = float(beta @ X.T @ X @ beta / (alpha * (1 - alpha)))
    return dq, float(stats.chi2.sf(dq, X.shape[1]))


def pinball_loss(returns: ArrayLike, q: ArrayLike, alpha: float) -> float:
    """Mean quantile (tick) loss; the strictly consistent scoring rule for quantiles."""
    _check_alpha(alpha)
    r, qv = np.asarray(returns, dtype=float), np.asarray(q, dtype=float)
    u = r - qv
    return float(np.mean(np.where(u < 0, (alpha - 1) * u, alpha * u)))


def basel_zones(hits: ArrayLike, window: int = 250) -> dict[str, float]:
    """Share of rolling ``window``-day periods in each zone of the Basel traffic light.

    The zones are defined for 99% VaR over 250 days: green up to 4 exceptions, yellow 5 to 9,
    red 10 or more.
    """
    h = _as_hits(hits)
    if len(h) < window:
        raise ValueError("need at least one full window")
    counts = np.convolve(h, np.ones(window, dtype=int), mode="valid")
    return {
        "green": float(np.mean(counts <= 4)),
        "yellow": float(np.mean((counts >= 5) & (counts <= 9))),
        "red": float(np.mean(counts >= 10)),
        "max_exceptions": float(counts.max()),
    }


@dataclass(frozen=True)
class BacktestResult:
    n: int
    alpha: float
    hit_rate: float
    p_kupiec: float
    p_independence: float
    p_conditional: float
    p_dq: float
    pinball: float
    mean_var: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def backtest(returns: ArrayLike, q: ArrayLike, alpha: float) -> BacktestResult:
    """Run every test on aligned returns and quantile forecasts (no missing values)."""
    r, qv = np.asarray(returns, dtype=float), np.asarray(q, dtype=float)
    if r.shape != qv.shape:
        raise ValueError("returns and q must have the same length")
    if np.isnan(r).any() or np.isnan(qv).any():
        raise ValueError("returns and q must not contain NaN")
    hits = (r < qv).astype(int)
    _, p_uc = kupiec(hits, alpha)
    cc = christoffersen(hits, alpha)
    _, p_dq = dynamic_quantile(hits, qv, alpha)
    return BacktestResult(
        n=len(r),
        alpha=alpha,
        hit_rate=float(hits.mean()),
        p_kupiec=p_uc,
        p_independence=cc["p_ind"],
        p_conditional=cc["p_cc"],
        p_dq=p_dq,
        pinball=pinball_loss(r, qv, alpha),
        mean_var=float(-qv.mean()),
    )
