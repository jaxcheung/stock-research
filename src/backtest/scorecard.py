# src/backtest/scorecard.py
#
# The app's CFA-style fundamental scorecard (investment_score.score_business), rebuilt
# POINT-IN-TIME so it can be backtested. At each rebalance date t, every eligible stock is
# scored with exactly the same bands and weights the research report uses, from:
#   quality          ROE (net income / ending equity; n/m if equity <= 0) and net margin
#   growth           revenue and net income growth vs the prior FULL fiscal year
#   financialHealth  current ratio, debt/equity, interest coverage (EBIT / interest expense);
#                    equity / assets for banks, lenders and broker-dealers; unscored for
#                    insurers and captive-finance groups (same rules as the live scorecard)
#   momentum         month-end close vs its 50- and 200-trading-day moving averages
# using only statements public by t (data.fundamentals_asof) and prices up to t.
# The business model comes from business_model.classify on that year's statements, with
# TODAY's FMP sector/industry labels (caveat). Valuation is NOT part of the score (CLAUDE.md
# rule): the "value-filtered" strategy applies a separate, disclosed E/P screen.
#
# Differences from the live scorecard, all disclosed: annual statements instead of FMP's
# ratio endpoints and TTM figures, and moving averages from the stored daily closes.

from typing import Any

import numpy as np
import pandas as pd

from src.analysis.business_model import classify, scorecard_revenue
from src.analysis.investment_score import score_business
from src.backtest.strategies import Context, _full_year, _usd_reporting
from src.config import (
    BACKTEST_MIN_UNIVERSE,
    BACKTEST_SCORECARD_MIN_GROWTH_COVERAGE,
    DCF_RISK_FREE_RATE,
    SCORE_RATING_BANDS,
)
from src.utils.logging import get_logger

logger = get_logger("backtest.scorecard")

DIMENSIONS = ("quality", "growth", "financialHealth", "momentum")


def _f(v: Any) -> float | None:
    """Plain float or None (NaN / missing -> None, as the scorecard expects)."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _ratio(a: Any, b: Any, positive_denominator: bool = True) -> float | None:
    x, y = _f(a), _f(b)
    if x is None or y is None or y == 0 or (positive_denominator and y < 0):
        return None
    return x / y


def _growth(cur: Any, prev: Any) -> float | None:
    c, p = _f(cur), _f(prev)
    if c is None or p is None or p <= 0:
        return None  # n/m from a zero or negative base — same rule as stock_analysis._growth
    return (c - p) / p


def stock_inputs(sym: str, row: pd.Series, ctx: Context) -> dict[str, Any]:
    """The `stock` dict score_business expects, from one point-in-time fundamentals row."""
    d = ctx.data
    t = ctx.t
    price = _f(d.close.at[t, sym]) if sym in d.close.columns else None
    ma50 = _f(d.ma50.at[t, sym]) if sym in d.ma50.columns and t in d.ma50.index else None
    ma200 = _f(d.ma200.at[t, sym]) if sym in d.ma200.columns and t in d.ma200.index else None
    equity = _f(row.get("equity"))
    if equity == 0:
        equity = None  # zero equity is a data gap (KKR FY2017), not negative equity
    profile = {
        "symbol": sym,
        "sector": d.sectors.get(sym),
        "industry": d.industries.get(sym),
        "marketCap": _f(ctx.market_cap.get(sym)),
    }
    income = {
        k: _f(row.get(k))
        for k in ("revenue", "interestIncome", "interestExpense", "netInterestIncome", "netIncome")
    }
    balance = {
        "totalAssets": _f(row.get("totalAssets")),
        "totalStockholdersEquity": equity,
        "totalDebt": _f(row.get("totalDebt")),
        "netReceivables": _f(row.get("netReceivables")),
        "accountPayables": _f(row.get("accountPayables")),
        "capitalLeaseObligationsCurrent": _f(row.get("leasesCurrent")),
        "capitalLeaseObligationsNonCurrent": _f(row.get("leasesNonCurrent")),
    }
    # Business model is a STRUCTURAL label: the captive-finance cost-of-debt test is run against the
    # static long-run rf assumption, not the T-bill of the day — with T-bills near 0 (2020-21) the
    # test can't fire and F/GM flipped between captive and standard with the rate regime.
    bm = classify(profile, income, balance, risk_free_rate=DCF_RISK_FREE_RATE)
    # Same revenue basis as the live scorecard (net of interest expense for gross-reporting banks,
    # broker-dealers and card lenders), for BOTH years so growth compares like with like.
    rev = scorecard_revenue(income, bm.kind, bm.subtype)
    prior = {
        "revenue": _f(row.get("prevRevenue")),
        "interestIncome": _f(row.get("prevInterestIncome")),
        "interestExpense": _f(row.get("prevInterestExpense")),
    }
    rev_prior = scorecard_revenue(prior, bm.kind, bm.subtype) if prior["revenue"] is not None else None
    net_basis = rev is not None and rev != income["revenue"]
    prev_ni = _f(row.get("prevNetIncome"))
    interest = _f(row.get("interestExpense"))
    coverage = _ratio(row.get("operatingIncome"), interest) if interest and interest > 0 else None
    return {
        "symbol": sym,
        "companyName": d.names.get(sym, sym),
        "price": price,
        "businessModel": bm.kind,
        "captiveFinance": bm.captive_finance,
        "profitability": {
            "returnOnEquity": _ratio(row.get("netIncome"), equity),
            "netProfitMargin": _ratio(row.get("netIncome"), rev),
        },
        "growth": {
            "revenueGrowthYoY": _growth(rev, rev_prior),
            "netIncomeGrowthYoY": _growth(row.get("netIncome"), prev_ni),
            "revenueBasis": "net of interest expense" if net_basis else "reported",
            "priorNetIncomeNonPositive": prev_ni is not None and prev_ni <= 0,
        },
        "leverage": {
            "totalEquity": equity,
            "equityToAssets": _ratio(equity, row.get("totalAssets")),
            "currentRatio": _ratio(row.get("totalCurrentAssets"), row.get("totalCurrentLiabilities")),
            "debtToEquityRatio": _ratio(row.get("totalDebt"), equity, positive_denominator=False),
            "interestCoverageRatio": coverage,
        },
        "technical": {"priceAvg50": ma50, "priceAvg200": ma200},
    }


def scores_at(ctx: Context) -> pd.DataFrame:
    """Scorecard for every eligible stock at ctx.t (memoised per date on ctx.data):
    columns = the four dimension scores, composite, rating, businessModel, fiscal period used."""
    key = ("scorecard", ctx.t)
    memo = ctx.data.memo
    if key in memo:
        return memo[key]
    f = _full_year(ctx.fundamentals.reindex(ctx.universe))  # part-year stubs aren't annual results
    rows = []
    for sym, row in f.iterrows():
        try:
            inputs = stock_inputs(str(sym), row, ctx)
            dims, composite, rating = score_business(inputs)
        except Exception as exc:  # one bad company never kills a rebalance
            logger.warning("Scorecard failed for %s at %s: %s", sym, ctx.t.date(), exc)
            continue
        if composite is None:
            continue
        rows.append(
            {
                "symbol": sym,
                **{k: dims[k]["score"] for k in DIMENSIONS},
                "composite": composite,
                "rating": rating,
                "businessModel": inputs["businessModel"],
                # health is unscored BY DESIGN for insurers and captive finance; otherwise a missing
                # health score is a data gap, which must not become a ranking advantage
                "healthByDesign": inputs["businessModel"] == "insurer" or bool(inputs["captiveFinance"]),
                "fiscalPeriod": pd.Timestamp(row["periodEnd"]).strftime("%Y-%m"),
                "dimensionsScored": sum(dims[k]["score"] is not None for k in DIMENSIONS),
            }
        )
    out = pd.DataFrame(rows).set_index("symbol") if rows else pd.DataFrame(columns=["composite", "rating"])
    memo[key] = out
    return out


def _tiebreak(ctx: Context, idx: pd.Index) -> pd.Series:
    """Banded scores tie often; break ties by point-in-time market cap (larger first), a
    deterministic, disclosed rule that doesn't depend on symbol order."""
    cap = ctx.market_cap.reindex(idx).fillna(0.0)
    return cap.rank(pct=True) * 1e-3


def _eligible(sc: pd.DataFrame, min_dims: int) -> pd.DataFrame:
    """A composite resting on few dimensions isn't the scorecard: need `min_dims` scored AND growth
    (growth is missing only when two full filed years don't exist yet — renormalising it away
    tests a different model), and financial health unless it is unscored by design (insurer,
    captive finance) — renormalising a data gap lifted KKR FY2017 to a composite of 100."""
    if sc.empty:
        return sc
    health_ok = sc["financialHealth"].notna() | sc["healthByDesign"].astype(bool)
    return sc[(sc["dimensionsScored"] >= min_dims) & sc["growth"].notna() & health_ok]


def first_full_scorecard(data: Any, start: str | None = None) -> pd.Timestamp | None:
    """First month-end (>= start) where >= BACKTEST_SCORECARD_MIN_GROWTH_COVERAGE of the eligible
    universe has a growth score, i.e. the full four-dimension scorecard is in force."""
    from src.backtest.strategies import universe_at

    lo = pd.Timestamp(start) if start else None
    for t in pd.DatetimeIndex(data.returns.index):
        if lo is not None and t < lo:
            continue
        uni, fund, cap = universe_at(data, t)
        if len(uni) < BACKTEST_MIN_UNIVERSE:
            continue
        sc = scores_at(Context(t=t, data=data, universe=uni, fundamentals=fund, market_cap=cap))
        if not sc.empty and float(sc["growth"].notna().mean()) >= BACKTEST_SCORECARD_MIN_GROWTH_COVERAGE:
            return t
    return None


def scorecard_top(ctx: Context) -> pd.Series:
    """Highest composite fundamental score (needs >= 3 of 4 dimensions)."""
    sc = _eligible(scores_at(ctx), 3)
    if sc.empty:
        return pd.Series(dtype=float)
    return sc["composite"].astype(float) + _tiebreak(ctx, sc.index)


def scorecard_buy(ctx: Context) -> pd.Series:
    """Every stock the scorecard rates Buy or Strong Buy (composite >= the Buy band)."""
    buy = next(th for th, r in SCORE_RATING_BANDS if r == "Buy")
    sc = _eligible(scores_at(ctx), 3)
    sc = sc[sc["composite"] >= buy] if not sc.empty else sc
    if sc.empty:
        return pd.Series(dtype=float)
    return sc["composite"].astype(float) + _tiebreak(ctx, sc.index)


def scorecard_value(ctx: Context) -> pd.Series:
    """Buy-rated on fundamentals AND cheaper than the universe median on earnings yield.
    Two SEPARATE tests (rating vs valuation are never blended, CLAUDE.md): the score ranks,
    the E/P screen filters. USD reporters only for the E/P test (currency rule)."""
    top = scorecard_buy(ctx)
    if top.empty:
        return top
    f = _usd_reporting(_full_year(ctx.fundamentals.reindex(ctx.universe)))
    ep = (f["netIncome"] / ctx.market_cap.reindex(f.index)).replace([np.inf, -np.inf], np.nan).dropna()
    if ep.empty:
        return pd.Series(dtype=float)
    cheap = set(ep[ep >= ep.median()].index)
    return top[[s for s in top.index if s in cheap]]
