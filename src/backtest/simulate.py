# src/backtest/simulate.py
#
# Portfolio simulation on top of a backtest: turn monthly weights and returns into what an
# investor would have seen with real money — portfolio value in dollars, holdings at every
# rebalance (with the fundamental scores that put them there, for scorecard strategies), the
# trades made, sector allocation, and calendar-year returns vs the benchmark. Pure
# post-processing of BacktestResult: no new assumptions about returns or costs.

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.data import BacktestData
from src.backtest.engine import BacktestResult
from src.backtest.strategies import STRATEGIES, Context, universe_at

_TRADE_EPS = 1e-4  # weight changes below 1bp are drift noise, not trades


@dataclass
class Simulation:
    strategy: str
    capital: float
    value: pd.Series  # portfolio value ($) at each month-end, starting at `capital`
    benchmark_value: pd.Series  # same capital in the benchmark
    holdings: dict[pd.Timestamp, pd.DataFrame]  # rebalance date -> positions after trading
    trades: pd.DataFrame  # date, symbol, action, weight before / after, $ traded
    sectors: pd.DataFrame  # rebalance date x sector -> weight (cash included)
    annual: pd.DataFrame  # calendar year -> strategy, benchmark, difference
    notes: list[str] = field(default_factory=list)


def _value(returns: pd.Series, capital: float, start: pd.Timestamp) -> pd.Series:
    v = capital * (1 + returns).cumprod()
    return pd.concat([pd.Series({start: capital}), v])


def simulate(
    result: BacktestResult,
    data: BacktestData,
    benchmark: BacktestResult,
    capital: float = 100_000.0,
) -> Simulation:
    if not np.isfinite(capital) or capital <= 0:
        raise ValueError(f"capital must be a positive amount, got {capital}")
    if not result.weights:
        raise ValueError("run_backtest(..., keep_weights=True) is required for a simulation")
    dates = sorted(result.weights)
    start = dates[0]
    value = _value(result.returns, capital, start)
    bench = _value(benchmark.returns.reindex(result.returns.index).fillna(0.0), capital, start)

    strat = STRATEGIES.get(result.strategy)
    holdings: dict[pd.Timestamp, pd.DataFrame] = {}
    trade_rows: list[dict[str, Any]] = []
    sector_rows: dict[pd.Timestamp, pd.Series] = {}
    for t in dates:
        w = result.weights[t]
        before = result.drifted.get(t, pd.Series(dtype=float))
        nav = float(value.get(t, np.nan))
        h = pd.DataFrame({"weight": w})
        h["value"] = h["weight"] * nav
        h["name"] = [data.names.get(s, "") for s in h.index]
        h["sector"] = [data.sectors.get(s, "ETF" if s in data.etfs else "Unknown") for s in h.index]
        if strat is not None and strat.name.startswith("scorecard"):
            from src.backtest.scorecard import DIMENSIONS, scores_at

            uni, fund, cap = universe_at(data, t)
            sc = scores_at(Context(t=t, data=data, universe=uni, fundamentals=fund, market_cap=cap))
            cols = [*DIMENSIONS, "composite", "rating", "fiscalPeriod"]
            h = h.join(sc.reindex(h.index)[cols]) if not sc.empty else h
        holdings[t] = h.sort_values("weight", ascending=False)
        cash = max(0.0, 1.0 - float(w.sum()))
        sec = h.groupby("sector")["weight"].sum()
        if cash > 1e-6:
            sec["Cash"] = cash
        sector_rows[t] = sec
        names = w.index.union(before.index)
        for s in names:
            a, b = float(before.get(s, 0.0)), float(w.get(s, 0.0))
            if abs(b - a) < _TRADE_EPS:
                continue
            action = "BUY" if a == 0 else "SELL" if b == 0 else "ADD" if b > a else "TRIM"
            trade_rows.append(
                {
                    "date": t,
                    "symbol": s,
                    "name": data.names.get(s, ""),
                    "action": action,
                    "weightBefore": a,
                    "weightAfter": b,
                    "amount": (b - a) * nav,
                }
            )
    trades = pd.DataFrame(
        trade_rows, columns=["date", "symbol", "name", "action", "weightBefore", "weightAfter", "amount"]
    )
    sectors = pd.DataFrame(sector_rows).T.fillna(0.0).sort_index()

    yr = pd.DataFrame(
        {"strategy": result.returns, "benchmark": benchmark.returns.reindex(result.returns.index)}
    )
    annual = (1 + yr).groupby(pd.DatetimeIndex(yr.index).year).prod() - 1
    months = yr.groupby(pd.DatetimeIndex(yr.index).year).size()
    annual["difference"] = annual["strategy"] - annual["benchmark"]
    annual["months"] = months
    notes = []
    partial = [str(y) for y, n in months.items() if n < 12]
    if partial:
        notes.append(f"Partial calendar years (fewer than 12 months): {', '.join(partial)}.")
    return Simulation(result.strategy, capital, value, bench, holdings, trades, sectors, annual, notes)
