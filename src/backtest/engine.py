# src/backtest/engine.py
#
# Monthly, long-only portfolio simulation.
#   At each month-end t: score eligible instruments with data available at t, build target
#   weights (top N, equal or inverse-volatility, position cap, sector cap), pay transaction
#   costs on the turnover from the DRIFTED previous weights, then earn the next month's total
#   return. Uninvested weight (e.g. dual momentum in cash) earns the risk-free rate.
#   A held stock with no return next month (data gap / stopped trading) earns 0 and is
#   counted in `missingReturns` (survivorship caveat: delisted names are not in the store).

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.backtest.data import BacktestData
from src.backtest.strategies import Context, Strategy, universe_at
from src.config import (
    BACKTEST_COST_BPS,
    BACKTEST_ETF_MAX_WEIGHT,
    BACKTEST_MAX_WEIGHT,
    BACKTEST_MIN_UNIVERSE,
    BACKTEST_SECTOR_CAP,
    BACKTEST_TOP_N,
)
from src.utils.logging import get_logger

logger = get_logger("backtest.engine")


@dataclass
class BacktestResult:
    strategy: str
    kind: str
    returns: pd.Series  # net monthly return, indexed by the month it was EARNED (t+1)
    gross: pd.Series
    turnover: pd.Series
    holdings: pd.Series  # number of positions
    weights: dict[pd.Timestamp, pd.Series] = field(default_factory=dict)
    missing_returns: int = 0
    notes: list[str] = field(default_factory=list)


def _cap_weights(w: pd.Series, cap: float) -> pd.Series:
    """Water-fill: clip at `cap`, redistribute the excess pro rata to uncapped names."""
    w = w / w.sum()
    for _ in range(50):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = float((w[over] - cap).sum())
        w[over] = cap
        under = ~over & (w < cap)
        if not under.any():
            break
        w[under] += excess * w[under] / w[under].sum()
    return w


def _sector_cap(w: pd.Series, sectors: pd.Series, cap: float) -> pd.Series:
    sec = sectors.reindex(w.index).fillna("Unknown")
    for _ in range(50):
        tot = w.groupby(sec).sum()
        over = tot[tot > cap + 1e-12]
        if over.empty:
            break
        for s, v in over.items():
            w[sec == s] *= cap / v
        free = ~sec.isin(over.index)
        if not free.any():
            break
        w[free] *= (1 - float(w[~free].sum())) / float(w[free].sum())
    return w


def build_weights(scores: pd.Series, ctx: Context, strategy: Strategy, top_n: int | None = None) -> pd.Series:
    scores = scores.replace([np.inf, -np.inf], np.nan).dropna()
    if scores.empty:
        return pd.Series(dtype=float)
    n = strategy.top_n or top_n or BACKTEST_TOP_N
    picks = scores.sort_values(ascending=False).iloc[:n].index
    if strategy.weighting == "inverse_vol":
        vol = ctx.history(12)[list(picks)].std(ddof=1).replace(0, np.nan)
        w = (1 / vol).dropna()
    else:
        w = pd.Series(1.0, index=picks)
    if w.empty:
        return w
    if strategy.kind == "stock":
        w = _cap_weights(w, max(BACKTEST_MAX_WEIGHT, 1.0 / len(w)))
        w = _sector_cap(w, ctx.data.sectors, BACKTEST_SECTOR_CAP)
    else:
        w = _cap_weights(w, max(BACKTEST_ETF_MAX_WEIGHT, 1.0 / len(w)))
    return w / w.sum()


def run_backtest(
    strategy: Strategy,
    data: BacktestData,
    start: str | None = None,
    end: str | None = None,
    cost_bps: float = BACKTEST_COST_BPS,
    top_n: int | None = None,
    keep_weights: bool = False,
) -> BacktestResult:
    if cost_bps < 0:
        raise ValueError(f"cost_bps must be >= 0, got {cost_bps}")
    lo = pd.Timestamp(start) if start else None
    hi = pd.Timestamp(end) if end else None
    dates = pd.DatetimeIndex(
        [
            d
            for d in pd.DatetimeIndex(data.returns.index)
            if (lo is None or d >= lo) and (hi is None or d <= hi)
        ]
    )
    if len(dates) < 3:
        raise ValueError("Backtest window needs at least 3 month-ends")
    net, gross, turn, hold = {}, {}, {}, {}
    weights: dict[pd.Timestamp, pd.Series] = {}
    prev = pd.Series(dtype=float)  # weights after last month's drift
    missing = 0
    thin_months: list[str] = []
    thin_signal: list[str] = []
    for t, t_next in zip(dates[:-1], dates[1:], strict=True):
        if strategy.kind == "stock":
            uni, fund, cap = universe_at(data, t)
            if len(uni) < BACKTEST_MIN_UNIVERSE:
                thin_months.append(t.strftime("%Y-%m"))
        else:
            uni, fund, cap = [], pd.DataFrame(), pd.Series(dtype=float)
        ctx = Context(t=t, data=data, universe=uni, fundamentals=fund, market_cap=cap)
        scores = strategy.signal(ctx)
        want = strategy.top_n or top_n or BACKTEST_TOP_N
        if (
            strategy.kind == "stock"
            and len(uni) >= BACKTEST_MIN_UNIVERSE
            and scores.dropna().size < min(want, len(uni))
        ):
            thin_signal.append(t.strftime("%Y-%m"))
        target = build_weights(scores, ctx, strategy, top_n)
        if target.empty and not strategy.cash_when_empty:
            target = prev.copy()  # nothing to do this month: hold (drifted) positions
        names = target.index.union(prev.index)
        to = float((target.reindex(names, fill_value=0) - prev.reindex(names, fill_value=0)).abs().sum())
        r_next = data.returns.loc[t_next].reindex(target.index)
        missing += int(r_next.isna().sum())
        r_next = r_next.fillna(0.0)
        cash = max(0.0, 1.0 - float(target.sum()))
        rf = float(data.rf.get(t_next, 0.0))
        g = float((target * r_next).sum()) + cash * rf
        cost = to * cost_bps / 1e4
        net[t_next], gross[t_next], turn[t_next], hold[t_next] = g - cost, g, to, len(target)
        if keep_weights:
            weights[t] = target
        grown = target * (1 + r_next)
        total = float(grown.sum()) + cash * (1 + rf)
        prev = pd.Series(grown / total) if total > 0 else pd.Series(dtype=float)
    res = BacktestResult(
        strategy.name,
        strategy.kind,
        pd.Series(net),
        pd.Series(gross),
        pd.Series(turn),
        pd.Series(hold),
        weights,
        missing,
    )
    if thin_months:
        res.notes.append(
            f"{len(thin_months)} rebalance month(s) had fewer than {BACKTEST_MIN_UNIVERSE} eligible stocks "
            f"({thin_months[0]}–{thin_months[-1]}); the strategy was partly or fully in cash then — results over "
            "that window are not a test of the signal."
        )
    if thin_signal:
        res.notes.append(
            f"{len(thin_signal)} month(s) where the signal covered fewer stocks than the {strategy.top_n or top_n or BACKTEST_TOP_N} "
            f"it should hold ({thin_signal[0]}–{thin_signal[-1]}), so the portfolio was more concentrated — e.g. growth "
            "needs two filed years of revenue."
        )
    if missing:
        res.notes.append(
            f"{missing} position-months had no next-month return (data gap or stopped trading) and were "
            "assumed flat — delisted companies are absent from the store (survivorship bias)."
        )
    return res
