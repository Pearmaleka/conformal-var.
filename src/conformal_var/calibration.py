"""Online calibration of quantile forecasts.

A base model forecasts the alpha-quantile ``q_hat_t`` of tomorrow's return. Calibration moves it
by a correction ``c_t`` computed from past data only, ``q_t = q_hat_t - c_t``. With the score
``s_t = q_hat_t - r_t``, an exception ``r_t < q_t`` happens exactly when ``s_t > c_t``, so
targeting an exception rate of alpha means tracking the (1 - alpha)-quantile of the scores.

Methods
-------
``none``
    The base forecast as is.
``rolling``
    Split conformal on a trailing window: ``c_t`` is the finite-sample-corrected
    (1 - alpha)-quantile of the last ``window`` scores. Valid if scores are exchangeable, which
    returns are not.
``aci``
    Adaptive conformal inference (Gibbs and Candes, 2021). The quantile level itself is
    adjusted after every day, ``alpha_{t+1} = alpha_t + gamma (alpha - err_t)``, which
    guarantees ``|mean(err) - alpha| <= (max(alpha, 1 - alpha) + gamma) / (gamma T)`` for any
    sequence of returns.
``tracking``
    Online quantile tracking, the P part of conformal PID control (Angelopoulos, Candes and
    Tibshirani, 2023): ``c_{t+1} = c_t + eta_t (err_t - alpha)``, with a step proportional to
    the largest recent score so that it does not depend on the units of the returns.

References
----------
Gibbs, I. and Candes, E. (2021). Adaptive conformal inference under distribution shift.
    NeurIPS 34.
Angelopoulos, A., Candes, E. and Tibshirani, R. (2023). Conformal PID control for time series
    prediction. NeurIPS 36.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["METHODS", "Calibrated", "calibrate"]

Method = Literal["none", "rolling", "aci", "tracking"]
METHODS: tuple[Method, ...] = ("none", "rolling", "aci", "tracking")


@dataclass(frozen=True)
class Calibrated:
    """Calibrated forecasts, aligned with the inputs. NaN where no forecast was made."""

    q: np.ndarray
    correction: np.ndarray
    alpha_t: np.ndarray  # quantile level used each day (ACI); alpha for the other methods


def _conformal_quantile(scores: np.ndarray, level: float) -> float:
    """Finite-sample conformal quantile: the ceil((n + 1) level)-th smallest score."""
    n = len(scores)
    if level <= 0:
        return -math.inf
    k = math.ceil((n + 1) * level)
    if k > n:
        return math.inf
    return float(np.partition(scores, k - 1)[k - 1])


def calibrate(
    base_q: ArrayLike,
    returns: ArrayLike,
    alpha: float,
    method: Method = "aci",
    *,
    window: int = 500,
    gamma: float = 0.005,
    lr: float = 0.01,
    scale_lr: bool = True,
    cap_at_max: bool = True,
) -> Calibrated:
    """Calibrate a sequence of quantile forecasts online.

    Parameters
    ----------
    base_q, returns
        Aligned arrays: ``base_q[t]`` must use information up to ``t - 1`` only. NaN in
        ``base_q`` marks days without a forecast; they are skipped.
    alpha
        Target exception rate.
    method
        One of ``METHODS``.
    window
        Number of past scores used by ``rolling`` and ``aci`` (and to scale the step of
        ``tracking``). No calibrated forecast is made until this many scores exist.
    gamma
        ACI step size.
    lr, scale_lr
        Quantile-tracking step. With ``scale_lr`` the step is ``lr`` times the largest
        absolute score in the window; without, it is ``lr`` in the units of the returns.
    cap_at_max
        ACI asks for an infinite correction when ``alpha_t`` falls to about ``1 / window``, which
        as a VaR means minus infinity. With ``cap_at_max`` the largest score in the window is
        used instead. The coverage bound is exact only with ``cap_at_max=False``.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if window < 1 or gamma <= 0 or lr <= 0:
        raise ValueError("window, gamma and lr must be positive")
    bq = np.asarray(base_q, dtype=float)
    r = np.asarray(returns, dtype=float)
    if bq.shape != r.shape or bq.ndim != 1:
        raise ValueError("base_q and returns must be 1-d arrays of the same length")
    if np.isnan(r[~np.isnan(bq)]).any():
        raise ValueError("returns must be known wherever base_q is")

    n = len(r)
    q = np.full(n, np.nan)
    correction = np.full(n, np.nan)
    alpha_t = np.full(n, np.nan)
    if method == "none":
        ok = ~np.isnan(bq)
        q[ok], correction[ok], alpha_t[ok] = bq[ok], 0.0, alpha
        return Calibrated(q, correction, alpha_t)

    scores = np.empty(n)
    m = 0  # number of scores seen
    a = alpha  # ACI level
    c_track = math.nan  # quantile-tracking state
    for t in range(n):
        if math.isnan(bq[t]):
            continue
        if m >= window:
            past = scores[m - window : m]
            if method == "rolling":
                c = _conformal_quantile(past, 1 - alpha)
            elif method == "aci":
                c = _conformal_quantile(past, 1 - a) if a < 1 else -math.inf
                if c == math.inf and cap_at_max:
                    c = float(np.max(past))
            else:
                if math.isnan(c_track):  # warm start from the conformal quantile
                    c_track = _conformal_quantile(past, 1 - alpha)
                    if math.isinf(c_track):
                        c_track = float(np.max(past))
                c = c_track
            correction[t] = c
            q[t] = bq[t] - c
            alpha_t[t] = a if method == "aci" else alpha
            s = bq[t] - r[t]
            err = 1.0 if s > c else 0.0
            if method == "aci":
                a += gamma * (alpha - err)
            elif method == "tracking":
                step = lr * float(np.max(np.abs(past))) if scale_lr else lr
                c_track += step * (err - alpha)
        scores[m] = bq[t] - r[t]
        m += 1
    return Calibrated(q, correction, alpha_t)
