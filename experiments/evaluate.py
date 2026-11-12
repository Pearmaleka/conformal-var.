"""Backtest every base model with every calibration method on four assets.

For each asset and VaR level (1% and 5%), four base models produce one-day-ahead quantile
forecasts and four calibration methods adjust them online. All 16 combinations are evaluated
on the same days: from the first day on which every combination has a forecast to the end of
the data. Hyperparameters are the defaults of ``calibrate`` (window 500, ACI gamma 0.005,
tracking step 0.01 times the largest recent score); ``--sensitivity`` also varies the two step
sizes for the GARCH model.

Run from the repository root:  python experiments/evaluate.py [--quick] [--sensitivity]
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _common import save_figure, style_axes, write_csv, write_json

from conformal_var import (
    ASSETS,
    FORECASTERS,
    METHODS,
    backtest,
    basel_zones,
    calibrate,
    load_returns,
)

ALPHAS = (0.01, 0.05)
GAMMAS = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05)
LRS = (0.001, 0.003, 0.01, 0.03, 0.1)


def evaluate_asset(name: str, quick: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    r = load_returns(name)
    if quick:
        r = r.iloc[-3000:]
    rows: list[dict[str, Any]] = []
    paths: dict[str, Any] = {"returns": r}
    for alpha in ALPHAS:
        base = {b: f(r, alpha).to_numpy() for b, f in FORECASTERS.items()}
        q = {(b, m): calibrate(base[b], r.to_numpy(), alpha, m).q for b in base for m in METHODS}
        ok = np.logical_and.reduce([~np.isnan(v) for v in q.values()])
        start = int(np.argmax(ok))
        assert ok[start:].all()
        ret = r.to_numpy()[start:]
        for (b, m), qv in q.items():
            res = backtest(ret, qv[start:], alpha).as_dict()
            zones = basel_zones((ret < qv[start:]).astype(int))
            rows.append(
                {
                    "asset": name,
                    "base": b,
                    "method": m,
                    "first_day": r.index[start],
                    "last_day": r.index[-1],
                    **res,
                    "basel_red_share": zones["red"] if alpha == 0.01 else np.nan,
                    "max_exceptions_250d": zones["max_exceptions"],
                }
            )
        paths[alpha] = {"start": start, "q": q, "base": base}
    return rows, paths


def sensitivity(name: str, quick: bool) -> list[dict[str, Any]]:
    r = load_returns(name)
    if quick:
        r = r.iloc[-3000:]
    rows = []
    for alpha in ALPHAS:
        base = FORECASTERS["garch"](r, alpha).to_numpy()
        variants = {("aci", g): {"gamma": g} for g in GAMMAS}
        variants |= {("tracking", lr): {"lr": lr} for lr in LRS}
        q = {k: calibrate(base, r.to_numpy(), alpha, k[0], **kw).q for k, kw in variants.items()}
        ok = np.logical_and.reduce([~np.isnan(v) for v in q.values()])
        start = int(np.argmax(ok))
        for (method, value), qv in q.items():
            res = backtest(r.to_numpy()[start:], qv[start:], alpha)
            rows.append({"asset": name, "method": method, "step": value, **res.as_dict()})
    return rows


def plot_exceptions(paths: dict[str, Any]) -> None:
    """S&P 500 around 2008: 99% VaR of GARCH-t with and without ACI, and their exceptions."""
    r = paths["returns"]
    info = paths[0.01]
    series = {}
    for method in ("none", "aci"):
        q = pd.Series(info["q"][("garch", method)], index=r.index).iloc[info["start"] :]
        hits = (r.iloc[info["start"] :] < q).astype(float)
        series[method] = (q, hits.rolling(250).sum())
    window = slice("2007-01-01", "2012-12-31")
    rw = r.loc[window]

    def draw(theme: dict[str, Any]) -> plt.Figure:
        fig, (top, bottom) = plt.subplots(
            2, 1, figsize=(7.6, 5.0), sharex=True, gridspec_kw={"height_ratios": [1.15, 1]}
        )
        raw_c, aci_c = theme["series"][1], theme["series"][0]
        for ax in (top, bottom):
            style_axes(ax, theme)
            ax.grid(axis="x", visible=False)
        top.vlines(rw.index, 0, rw.to_numpy(), color=theme["axis"], lw=0.6)
        for method, color, label in (("none", raw_c, "GARCH-t"), ("aci", aci_c, "with ACI")):
            q = series[method][0].loc[window]
            top.plot(q.index, q.to_numpy(), color=color, lw=1.3, label=label)
        top.set_ylim(-16, 9)
        top.set_ylabel("Daily return and VaR (%)")
        leg = top.legend(loc="lower right", frameon=False, ncol=2)
        for text in leg.get_texts():
            text.set_color(theme["ink2"])
        bottom.axhspan(4.5, 9.5, color=theme["grid"], alpha=0.45, lw=0)
        bottom.axhspan(9.5, 14, color=theme["grid"], lw=0)
        left = pd.Timestamp("2007-02-01")
        bottom.text(left, 9.8, "Basel red zone", fontsize=8.5, color=theme["muted"])
        bottom.text(left, 4.8, "yellow zone", fontsize=8.5, color=theme["muted"])
        for method, color in (("none", raw_c), ("aci", aci_c)):
            c = series[method][1].loc[window]
            bottom.plot(c.index, c.to_numpy(), color=color, lw=1.8)
        bottom.set_ylim(0, 14)
        bottom.set_ylabel("Exceptions, past 250 days")
        bottom.xaxis.set_major_locator(mdates.YearLocator())
        bottom.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        top.set_title("S&P 500, 99% one-day VaR: fewer exceptions for a higher VaR", loc="left")
        fig.align_ylabels((top, bottom))
        fig.tight_layout()
        return fig

    save_figure(draw, "exceptions-sp500")


def plot_hit_rates(results: pd.DataFrame) -> None:
    d = results[(results["alpha"] == 0.01) & results["method"].isin(["none", "aci"])]
    bases = list(FORECASTERS)
    labels = {
        "hs": "Historical simulation",
        "ewma": "EWMA normal",
        "garch": "GARCH-t",
        "gbm": "Gradient boosting",
    }

    def draw(theme: dict[str, Any]) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(7.4, 2.9))
        style_axes(ax, theme)
        ax.grid(axis="y", visible=False)
        for i, b in enumerate(reversed(bases)):
            raw = d[(d["base"] == b) & (d["method"] == "none")]["hit_rate"].to_numpy() * 100
            aci = d[(d["base"] == b) & (d["method"] == "aci")]["hit_rate"].to_numpy() * 100
            for x0, x1 in zip(raw, aci, strict=True):
                ax.plot([x0, x1], [i, i], color=theme["axis"], lw=1, zorder=1)
            ax.scatter(
                raw,
                np.full(len(raw), i),
                s=46,
                color=theme["series"][1],
                zorder=3,
                edgecolors=theme["surface"],
                linewidths=1,
                label="Base model" if i == 0 else None,
            )
            ax.scatter(
                aci,
                np.full(len(aci), i),
                s=46,
                color=theme["series"][0],
                zorder=3,
                edgecolors=theme["surface"],
                linewidths=1,
                label="With ACI" if i == 0 else None,
            )
        ax.axvline(1.0, color=theme["muted"], lw=1, ls=(0, (3, 3)))
        ax.set_yticks(range(len(bases)), [labels[b] for b in reversed(bases)])
        ax.set_xlabel("Exception rate of 99% VaR (%), one dot per asset")
        ax.set_title("ACI brings every model to the target rate", loc="left")
        leg = ax.legend(loc="upper right", frameon=False, ncol=2)
        for text in leg.get_texts():
            text.set_color(theme["ink2"])
        return fig

    save_figure(draw, "hit-rates")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="last 3,000 days of each asset")
    parser.add_argument("--sensitivity", action="store_true")
    args = parser.parse_args()
    start = time.perf_counter()
    rows, sp500_paths = [], None
    for name in ASSETS:
        asset_rows, paths = evaluate_asset(name, args.quick)
        rows += asset_rows
        if name == "sp500":
            sp500_paths = paths
        print(f"{name}: {time.perf_counter() - start:.0f}s", flush=True)
    results = pd.DataFrame(rows)
    suffix = "-quick" if args.quick else ""
    write_csv(results, f"evaluation{suffix}.csv")
    meta: dict[str, Any] = {"quick": args.quick, "seconds": time.perf_counter() - start}
    if args.sensitivity:
        sens = pd.DataFrame([row for name in ASSETS for row in sensitivity(name, args.quick)])
        write_csv(sens, f"sensitivity{suffix}.csv")
    write_json(meta, f"evaluation_meta{suffix}.json")
    if not args.quick and sp500_paths is not None:
        plot_exceptions(sp500_paths)
        plot_hit_rates(results)
    cols = ["hit_rate", "p_kupiec", "p_conditional", "p_dq", "pinball", "mean_var"]
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(results.groupby(["alpha", "base", "method"])[cols].mean().round(4))
    print(f"{time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
