"""Daily return series used in the experiments.

* ``sp500``, ``nasdaq``: S&P 500 and NASDAQ Composite closes, 1999-2018, shipped with ``arch``.
* ``wti``: WTI crude oil spot price (FRED series DCOILWTICO), 1986-2019, shipped with ``arch``.
* ``usdjpy``: yen per US dollar (FRED, via github.com/datasets/exchange-rates at commit
  d33880bb076ce11576c4fe7e2ffd3bef904a98fe), 1971-2026, vendored in ``_data/``.
"""

from __future__ import annotations

from importlib import resources

import numpy as np
import pandas as pd

__all__ = ["ASSETS", "load_prices", "load_returns"]

ASSETS = {
    "sp500": "S&P 500",
    "nasdaq": "NASDAQ Composite",
    "wti": "WTI crude oil",
    "usdjpy": "USD/JPY",
}


def load_prices(name: str) -> pd.Series:
    """Daily prices, without missing days (holidays are dropped, not filled)."""
    if name in ("sp500", "nasdaq"):
        from arch.data import nasdaq, sp500

        module = sp500 if name == "sp500" else nasdaq
        prices = module.load()["Adj Close"]
    elif name == "wti":
        from arch.data import wti

        prices = wti.load()["DCOILWTICO"]
    elif name == "usdjpy":
        path = resources.files("conformal_var") / "_data" / "usdjpy_daily.csv"
        with resources.as_file(path) as p:
            frame = pd.read_csv(p, parse_dates=["date"], index_col="date")
        prices = frame["jpy_per_usd"]
    else:
        raise ValueError(f"unknown asset {name!r}; choose from {sorted(ASSETS)}")
    prices = prices.dropna().astype(float)
    prices.index = pd.DatetimeIndex(prices.index).rename("date")
    return prices.rename(name)


def load_returns(name: str) -> pd.Series:
    """Daily log returns in percent."""
    prices = load_prices(name)
    return (100 * np.log(prices).diff()).dropna().rename(name)
