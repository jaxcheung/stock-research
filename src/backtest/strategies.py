# src/backtest/strategies.py
#
# Rule-based strategies (no LLM). Each strategy maps a point-in-time Context to a score per
# eligible instrument (higher = more attractive); the engine turns scores into long-only
# weights. A Context only exposes data available at the rebalance date t:
#   - prices / returns up to and including t (month-end close)
#   - fundamentals whose availableFrom <= t (see data.py)
#   - the universe at t (market cap >= BACKTEST_MIN_CAP from close_t x latest filed shares, and
#     median daily dollar volume >= BACKTEST_MIN_DOLLAR_VOLUME; preferreds and ETFs excluded)
#
# Signals (academic standard definitions, monthly):
#   momentum_12_1   P(t-1)/P(t-12) - 1     (skip the latest month: short-term reversal)
#   low_vol         -stdev of the last 12 monthly returns
#   value_ep        latest annual net income / market cap      (earnings yield)
#   value_bp        latest annual book equity / market cap
#   quality_gp      gross profit / total assets                (Novy-Marx)
#   growth          latest annual revenue growth
#   multi_factor    mean of cross-sectional z-scores of value_ep, quality_gp, momentum, low_vol

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.backtest.data import BacktestData
from src.config import (
    BACKTEST_MIN_CAP,
    BACKTEST_MIN_DOLLAR_VOLUME,
    BACKTEST_MIN_HISTORY_MONTHS,
    BACKTEST_SECTOR_ETFS,
)
from src.data.sync import PREFERRED_SYMBOL


@dataclass
class Context:
    t: pd.Timestamp
    data: BacktestData
    universe: list[str]  # eligible stocks at t
    fundamentals: pd.DataFrame  # as-of t, indexed by symbol
    market_cap: pd.Series  # point-in-time cap at t, indexed by symbol

    def history(self, months: int) -> pd.DataFrame:
        """Monthly total returns for the `months` months ending at t (inclusive)."""
        r = self.data.returns.loc[: self.t]
        return r.iloc[-months:]


@dataclass(frozen=True)
class Strategy:
    name: str
    kind: str  # "stock" | "etf"
    signal: Callable[[Context], pd.Series]
    description: str
    weighting: str = "equal"  # "equal" | "inverse_vol"
    top_n: int | None = None  # None -> engine default for the kind
    cash_when_empty: bool = False  # all-cash (risk-free) when the signal returns nothing
    hold_all: bool = False  # hold EVERY name the signal returns (a rule, not a top-N ranking)


# --------------------------------------------------------------------------- universe


def universe_at(data: BacktestData, t: pd.Timestamp) -> tuple[list[str], pd.DataFrame, pd.Series]:
    """(eligible stocks, fundamentals as of t, market cap at t). Eligible = US-listed stock with
    enough price history, a price at t, and point-in-time cap >= BACKTEST_MIN_CAP. Memoised per
    date on `data` (every strategy in a suite sees the same universe)."""
    key = ("universe", t)
    if key not in data.memo:
        data.memo[key] = _universe_at(data, t)
    uni, fund, cap = data.memo[key]
    return list(uni), fund, cap


def _universe_at(data: BacktestData, t: pd.Timestamp) -> tuple[list[str], pd.DataFrame, pd.Series]:
    fund = data.fundamentals_asof(t)
    if t not in data.close.index:
        return [], fund, pd.Series(dtype=float)
    px = pd.Series(data.close.loc[t])
    shares = fund["shares"] if "shares" in fund else pd.Series(dtype=float)
    cap = pd.Series(px.reindex(shares.index) * shares).dropna()
    hist = data.adj.loc[:t].iloc[-BACKTEST_MIN_HISTORY_MONTHS:]
    enough = hist.notna().sum() >= BACKTEST_MIN_HISTORY_MONTHS
    etfs = set(data.etfs)
    liquid = (
        pd.Series(data.dollar_volume.loc[t]) >= BACKTEST_MIN_DOLLAR_VOLUME
        if not data.dollar_volume.empty and t in data.dollar_volume.index
        else pd.Series(True, index=cap.index)
    )
    eligible = [
        str(s)
        for s, c in cap.items()
        if c >= BACKTEST_MIN_CAP
        and s not in etfs
        and not PREFERRED_SYMBOL.search(str(s))  # preferred: issuer's shares x a $25 preferred price
        and bool(enough.get(s, False))
        and bool(liquid.get(s, False))
    ]
    return sorted(eligible), fund, cap


# --------------------------------------------------------------------------- signal helpers


def _zscore(s: pd.Series) -> pd.Series:
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 3:
        return s * np.nan
    lo, hi = s.quantile([0.01, 0.99])
    s = s.clip(lo, hi)
    sd = s.std(ddof=1)
    return (s - s.mean()) / sd if sd > 0 else s * 0.0


def _momentum(ctx: Context, names: list[str]) -> pd.Series:
    adj = ctx.data.adj.loc[: ctx.t, names]
    if len(adj) < 13:
        return pd.Series(dtype=float)
    return (adj.iloc[-2] / adj.iloc[-13] - 1).dropna()


def _volatility(ctx: Context, names: list[str]) -> pd.Series:
    return ctx.history(12)[names].std(ddof=1).dropna()


def momentum_12_1(ctx: Context) -> pd.Series:
    return _momentum(ctx, ctx.universe)


def low_vol(ctx: Context) -> pd.Series:
    return -_volatility(ctx, ctx.universe)


def _usd_reporting(f: pd.DataFrame) -> pd.DataFrame:
    """Currency guard (CLAUDE.md rule): price/market cap is in USD, so a statement-based value
    multiple is only meaningful when the company reports in USD. Non-USD reporters (CAD, GBP,
    CNY, ...) drop out of value signals; ratios within one currency (GP/assets, growth) stay."""
    if "currency" not in f:
        return f
    return f[f["currency"].fillna("USD").str.upper() == "USD"]


def _full_year(f: pd.DataFrame) -> pd.DataFrame:
    """Drop part-year transition periods: their flows (earnings, gross profit) aren't annual."""
    return f[f["fullYear"].fillna(True).astype(bool)] if "fullYear" in f else f


def value_ep(ctx: Context) -> pd.Series:
    f = _full_year(_usd_reporting(ctx.fundamentals.reindex(ctx.universe)))
    return (f["netIncome"] / ctx.market_cap.reindex(f.index)).dropna()


def value_bp(ctx: Context) -> pd.Series:
    f = _usd_reporting(ctx.fundamentals.reindex(ctx.universe))
    eq = f["equity"].where(f["equity"] > 0)  # negative book equity: no meaningful B/P
    return (eq / ctx.market_cap.reindex(f.index)).dropna()


def quality_gp(ctx: Context) -> pd.Series:
    f = _full_year(ctx.fundamentals.reindex(ctx.universe))
    return (f["grossProfit"] / f["totalAssets"].where(f["totalAssets"] > 0)).dropna()


def growth(ctx: Context) -> pd.Series:
    f = ctx.fundamentals.reindex(ctx.universe)
    return (f["revenue"] / f["prevRevenue"].where(f["prevRevenue"] > 0) - 1).dropna()


def multi_factor(ctx: Context) -> pd.Series:
    parts = [
        _zscore(value_ep(ctx)),
        _zscore(quality_gp(ctx)),
        _zscore(momentum_12_1(ctx)),
        _zscore(low_vol(ctx)),
    ]
    z = pd.concat(parts, axis=1)
    return z.mean(axis=1, skipna=True)[z.notna().sum(axis=1) >= 3]  # need 3 of 4 factors


def random_placebo(ctx: Context, seed: int = 0) -> pd.Series:
    """Placebo: random scores (seeded by seed and date). Must show no significant alpha."""
    rng = np.random.default_rng([seed, int(ctx.t.strftime("%Y%m"))])
    return pd.Series(rng.standard_normal(len(ctx.universe)), index=ctx.universe)


def make_placebo(seed: int) -> Strategy:
    return Strategy(
        f"placebo_{seed}",
        "stock",
        lambda ctx: random_placebo(ctx, seed),
        f"Random scores, seed {seed} (placebo)",
    )


def universe_equal_weight(ctx: Context) -> pd.Series:
    """Every eligible stock, equally weighted — the universe benchmark."""
    return pd.Series(1.0, index=ctx.universe)


# --------------------------------------------------------------------------- ETF strategies


def _sector_etfs(ctx: Context) -> list[str]:
    return [
        e for e in BACKTEST_SECTOR_ETFS if e in ctx.data.adj.columns and pd.notna(ctx.data.adj.at[ctx.t, e])
    ]


def sector_momentum(ctx: Context) -> pd.Series:
    """Top sector ETFs by 12-1 momentum."""
    return _momentum(ctx, _sector_etfs(ctx))


def sector_dual_momentum(ctx: Context) -> pd.Series:
    """Sector momentum, but only while SPY's 12-month return beats T-bills (absolute momentum);
    otherwise hold cash at the risk-free rate."""
    adj = ctx.data.adj.loc[: ctx.t]
    if "SPY" not in adj or len(adj) < 13:
        return pd.Series(dtype=float)
    spy = adj["SPY"].to_numpy(dtype=float)
    spy_12m = float(spy[-1] / spy[-13] - 1)
    rf_12m = float(np.prod(1 + ctx.data.rf.loc[: ctx.t].iloc[-12:].to_numpy(dtype=float)) - 1)
    if spy_12m <= rf_12m:
        return pd.Series(dtype=float)
    return sector_momentum(ctx)


def sector_equal_weight(ctx: Context) -> pd.Series:
    return pd.Series(1.0, index=_sector_etfs(ctx))


def spy_buy_hold(ctx: Context) -> pd.Series:
    return pd.Series(1.0, index=["SPY"]) if "SPY" in ctx.data.adj.columns else pd.Series(dtype=float)


def _scorecard(name: str) -> Callable[[Context], pd.Series]:
    """Signals from the point-in-time fundamental scorecard (imported lazily: scorecard.py
    builds on this module's Context)."""

    def signal(ctx: Context) -> pd.Series:
        from src.backtest import scorecard

        fn: Callable[[Context], pd.Series] = getattr(scorecard, name)
        return fn(ctx)

    return signal


STRATEGIES: dict[str, Strategy] = {
    s.name: s
    for s in [
        Strategy("momentum_12_1", "stock", momentum_12_1, "Top stocks by 12-1 month price momentum"),
        Strategy("low_vol", "stock", low_vol, "Lowest 12-month volatility stocks"),
        Strategy("value_ep", "stock", value_ep, "Highest earnings yield (net income / market cap)"),
        Strategy("value_bp", "stock", value_bp, "Highest book-to-price"),
        Strategy("quality_gp", "stock", quality_gp, "Highest gross profitability (gross profit / assets)"),
        Strategy("growth", "stock", growth, "Highest latest annual revenue growth"),
        Strategy("multi_factor", "stock", multi_factor, "Mean z-score of value, quality, momentum, low vol"),
        Strategy(
            "multi_factor_ivol",
            "stock",
            multi_factor,
            "Multi-factor, inverse-volatility weighted",
            weighting="inverse_vol",
        ),
        # The app's own CFA-style fundamental scorecard (quality, growth, financial health,
        # momentum — same bands and weights as the research report), point-in-time.
        Strategy(
            "scorecard_top",
            "stock",
            _scorecard("scorecard_top"),
            "Fundamental scorecard: highest composite score",
        ),
        Strategy(
            "scorecard_buy",
            "stock",
            _scorecard("scorecard_buy"),
            "Fundamental scorecard: hold every Buy / Strong Buy",
            hold_all=True,
        ),
        Strategy(
            "scorecard_value",
            "stock",
            _scorecard("scorecard_value"),
            "Fundamental scorecard Buy-rated AND E/P above the universe median (separate value screen)",
        ),
        Strategy("sector_momentum", "etf", sector_momentum, "Top 3 sector ETFs by 12-1 momentum", top_n=3),
        Strategy(
            "sector_dual_momentum",
            "etf",
            sector_dual_momentum,
            "Sector momentum, cash when SPY's 12m return < T-bills",
            top_n=3,
            cash_when_empty=True,
        ),
        # Reference portfolios (benchmarks / controls) — not counted as strategy trials.
        Strategy(
            "universe_ew",
            "stock",
            universe_equal_weight,
            "Equal-weight eligible universe (control)",
            hold_all=True,
        ),
        Strategy("sector_ew", "etf", sector_equal_weight, "Equal-weight 11 sector ETFs (control)", top_n=11),
        Strategy("spy", "etf", spy_buy_hold, "SPY buy-and-hold (benchmark)", top_n=1),
        Strategy("placebo_random", "stock", random_placebo, "Random scores (placebo; expect no alpha)"),
    ]
}
CONTROLS = ("universe_ew", "sector_ew", "spy")
# Never counted as a strategy trial: the placebo (run as many seeds by the lab).
NON_TRIALS = (*CONTROLS, "placebo_random")
