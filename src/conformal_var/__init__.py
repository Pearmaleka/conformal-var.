"""Value-at-Risk forecasts with online conformal calibration, and the backtests to judge them."""

from conformal_var.backtests import (
    BacktestResult,
    backtest,
    basel_zones,
    christoffersen,
    dynamic_quantile,
    kupiec,
    pinball_loss,
)
from conformal_var.calibration import METHODS, Calibrated, calibrate
from conformal_var.data import ASSETS, load_prices, load_returns
from conformal_var.forecasters import (
    FORECASTERS,
    ewma_normal,
    garch_t,
    gbm_quantile,
    historical_simulation,
)

__all__ = [
    "ASSETS",
    "FORECASTERS",
    "METHODS",
    "BacktestResult",
    "Calibrated",
    "backtest",
    "basel_zones",
    "calibrate",
    "christoffersen",
    "dynamic_quantile",
    "ewma_normal",
    "garch_t",
    "gbm_quantile",
    "historical_simulation",
    "kupiec",
    "load_prices",
    "load_returns",
    "pinball_loss",
]

__version__ = "0.1.0"
