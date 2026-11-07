# src/analysis/indicators.py
#
# Indicator catalog + per-company annual history, shared by the Stock
# Analysis and Industry Comparison pages.
#
# Every indicator belongs to one of three groups that are always reported
# separately:  "return" (profitability, growth, shareholder return, price
# return), "risk" (leverage/gearing, liquidity, earnings quality, price risk)
# and "valuation" (price-derived multiples).
#
# Rules applied here (CLAUDE.md):
# - Currency guard: a fiscal year whose reporting currency differs from the
#   quote currency has every price-derived indicator (and the market-cap /
#   EV raw columns behind aggregates) nulled. Indicators computed entirely
#   within the reporting currency (margins, ROE, gearing...) stay valid.
# - Aggregates of ratios are ratio-of-sums (Σ numerator ÷ Σ denominator), so
#   each indicator that has a meaningful aggregate names its raw numerator and
#   denominator columns in `agg`.
# - Partial data (no interest expense, negative equity, loss years) yields NaN
#   for that indicator-year rather than a misleading number or an exception.

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from src.analysis import timeseries as ts
from src.config import (
    INDICATOR_HISTORY_YEARS,
    INDUSTRY_INDICATOR_PROFILES,
    PRICE_HISTORY_YEARS,
    PRICE_RISK_WINDOW_YEARS,
)
from src.data.fmp_client import FMPClient, FMPError, normalize_symbol
from src.utils.logging import get_logger

logger = get_logger("indicators")

GROUPS = {"return": "Return", "risk": "Risk", "valuation": "Valuation"}
Fmt = str  # "pct" | "x" (multiple) | "ratio" (plain number) | "days"


@dataclass(frozen=True)
class Indicator:
    key: str
    label: str
    group: str  # "return" | "risk" | "valuation"
    category: str  # finer grouping for pickers: Growth, Profitability, Leverage, ...
    fmt: Fmt
    higher_is_better: bool | None  # None = neutral / context-dependent
    description: str
    compute: Callable[[pd.DataFrame], pd.Series] | None = None  # None = market (price) indicator
    agg: tuple[str, str] | None = None  # raw (numerator, denominator) for ratio-of-sums aggregate
    price_derived: bool = False  # nulled on quote/reporting currency mismatch
    positive_only: bool = False  # negative values are "not meaningful" (loss-making P/E ...)
    market: bool = False  # computed from daily prices, snapshot only (not per fiscal year)


# --------------------------------------------------------------------------- helpers


def _c(d: pd.DataFrame, name: str) -> pd.Series:
    if name in d:
        return pd.to_numeric(d[name], errors="coerce").astype(float)
    return pd.Series(np.nan, index=d.index, dtype=float)


def _div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.where(b != 0)


def _div_pos(a: pd.Series, b: pd.Series) -> pd.Series:
    """a ÷ b only where the denominator is positive (negative equity / EBITDA
    / earnings make the ratio's sign meaningless)."""
    return a / b.where(b > 0)


def _growth(s: pd.Series) -> pd.Series:
    """Year-over-year growth; NaN when the prior value is <= 0 (growth from a
    loss or zero base has no meaningful sign) or the prior year is missing."""
    prev = s.shift(1)
    consecutive = pd.Series(s.index, index=s.index).diff() == 1
    return ((s - prev) / prev.where(prev > 0)).where(consecutive)


# Raw columns used by the catalog. Prefixes: inc_ income statement, bs_
# balance sheet, cf_ cash flow, r_ FMP ratios, km_ FMP key metrics, x_ derived.
_I = "inc_"
_B = "bs_"
_F = "cf_"


def derive_raw(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["x_ffo"] = _c(d, "inc_netIncome") + _c(d, "inc_depreciationAndAmortization").fillna(0)
    d["x_netRevenue"] = _c(d, "inc_revenue") - _c(d, "inc_interestExpense").fillna(0)
    d["x_debtPlusEquity"] = _c(d, "bs_totalDebt") + _c(d, "bs_totalStockholdersEquity")
    d["x_tangibleEquity"] = _c(d, "bs_totalStockholdersEquity") - _c(
        d, "bs_goodwillAndIntangibleAssets"
    ).fillna(0)
    d["x_capex"] = _c(d, "cf_capitalExpenditure").abs()
    d["x_dividends"] = _c(d, "cf_commonDividendsPaid").abs()
    d["x_revenueGrowth"] = _growth(_c(d, "inc_revenue"))
    return d


def _m(col: str) -> Callable[[pd.DataFrame], pd.Series]:
    return lambda d: _c(d, col)


def _r(num: str, den: str, pos: bool = False) -> Callable[[pd.DataFrame], pd.Series]:
    f = _div_pos if pos else _div
    return lambda d: f(_c(d, num), _c(d, den))


_CATALOG: list[Indicator] = [
    # ---------------------------------------------------------------- return: growth
    Indicator(
        "revenueGrowth",
        "Revenue growth",
        "return",
        "Growth",
        "pct",
        True,
        "Year-over-year revenue growth.",
        _m("x_revenueGrowth"),
    ),
    Indicator(
        "epsGrowth",
        "EPS growth (diluted)",
        "return",
        "Growth",
        "pct",
        True,
        "Year-over-year diluted EPS growth; blank when prior-year EPS <= 0.",
        lambda d: _growth(_c(d, "inc_epsDiluted")),
    ),
    Indicator(
        "fcfGrowth",
        "FCF growth",
        "return",
        "Growth",
        "pct",
        True,
        "Year-over-year free cash flow growth; blank when prior-year FCF <= 0.",
        lambda d: _growth(_c(d, "cf_freeCashFlow")),
    ),
    # ---------------------------------------------------------------- return: profitability
    Indicator(
        "grossMargin",
        "Gross margin",
        "return",
        "Profitability",
        "pct",
        True,
        "Gross profit ÷ revenue.",
        _r("inc_grossProfit", "inc_revenue"),
        ("inc_grossProfit", "inc_revenue"),
    ),
    Indicator(
        "operatingMargin",
        "Operating margin",
        "return",
        "Profitability",
        "pct",
        True,
        "Operating income ÷ revenue.",
        _r("inc_operatingIncome", "inc_revenue"),
        ("inc_operatingIncome", "inc_revenue"),
    ),
    Indicator(
        "ebitdaMargin",
        "EBITDA margin",
        "return",
        "Profitability",
        "pct",
        True,
        "EBITDA ÷ revenue.",
        _r("inc_ebitda", "inc_revenue"),
        ("inc_ebitda", "inc_revenue"),
    ),
    Indicator(
        "netMargin",
        "Net margin",
        "return",
        "Profitability",
        "pct",
        True,
        "Net income ÷ revenue.",
        _r("inc_netIncome", "inc_revenue"),
        ("inc_netIncome", "inc_revenue"),
    ),
    Indicator(
        "fcfMargin",
        "FCF margin",
        "return",
        "Profitability",
        "pct",
        True,
        "Free cash flow (operating cash flow − capex) ÷ revenue.",
        _r("cf_freeCashFlow", "inc_revenue"),
        ("cf_freeCashFlow", "inc_revenue"),
    ),
    Indicator(
        "ffoMargin",
        "FFO margin (approx.)",
        "return",
        "Profitability",
        "pct",
        True,
        "Approximate funds from operations (net income + D&A, gains on sale not removed) ÷ revenue — REITs.",
        _r("x_ffo", "inc_revenue"),
        ("x_ffo", "inc_revenue"),
    ),
    Indicator(
        "roe",
        "Return on equity",
        "return",
        "Profitability",
        "pct",
        True,
        "Net income ÷ period-end shareholders' equity; blank when equity <= 0.",
        _r("inc_netIncome", "bs_totalStockholdersEquity", pos=True),
        ("inc_netIncome", "bs_totalStockholdersEquity"),
    ),
    Indicator(
        "roa",
        "Return on assets",
        "return",
        "Profitability",
        "pct",
        True,
        "Net income ÷ period-end total assets.",
        _r("inc_netIncome", "bs_totalAssets", pos=True),
        ("inc_netIncome", "bs_totalAssets"),
    ),
    Indicator(
        "roic",
        "Return on invested capital",
        "return",
        "Profitability",
        "pct",
        True,
        "FMP definition: after-tax operating income ÷ invested capital.",
        _m("km_returnOnInvestedCapital"),
    ),
    Indicator(
        "roce",
        "Return on capital employed",
        "return",
        "Profitability",
        "pct",
        True,
        "FMP definition: EBIT ÷ (total assets − current liabilities).",
        _m("km_returnOnCapitalEmployed"),
    ),
    Indicator(
        "netInterestShare",
        "Net interest income share",
        "return",
        "Banking",
        "pct",
        None,
        "Net interest income ÷ net revenue (revenue − interest expense) — banks' reliance on lending spread.",
        _r("inc_netInterestIncome", "x_netRevenue", pos=True),
        ("inc_netInterestIncome", "x_netRevenue"),
    ),
    Indicator(
        "efficiencyRatio",
        "Efficiency ratio",
        "return",
        "Banking",
        "pct",
        False,
        "Operating expenses ÷ net revenue (revenue − interest expense); lower is better — banks.",
        _r("inc_operatingExpenses", "x_netRevenue", pos=True),
        ("inc_operatingExpenses", "x_netRevenue"),
    ),
    Indicator(
        "ruleOf40",
        "Rule of 40",
        "return",
        "Growth",
        "pct",
        True,
        "Revenue growth + FCF margin — software growth/profitability balance (≥ 40% is the usual bar).",
        lambda d: _c(d, "x_revenueGrowth") + _div(_c(d, "cf_freeCashFlow"), _c(d, "inc_revenue")),
    ),
    Indicator(
        "assetTurnover",
        "Asset turnover",
        "return",
        "Efficiency",
        "ratio",
        True,
        "Revenue ÷ total assets.",
        _r("inc_revenue", "bs_totalAssets"),
        ("inc_revenue", "bs_totalAssets"),
    ),
    Indicator(
        "rdIntensity",
        "R&D intensity",
        "return",
        "Efficiency",
        "pct",
        None,
        "R&D expense ÷ revenue.",
        _r("inc_researchAndDevelopmentExpenses", "inc_revenue"),
        ("inc_researchAndDevelopmentExpenses", "inc_revenue"),
    ),
    Indicator(
        "payoutRatio",
        "Dividend payout ratio",
        "return",
        "Shareholder return",
        "pct",
        None,
        "Common dividends paid ÷ net income; blank when net income <= 0.",
        _r("x_dividends", "inc_netIncome", pos=True),
        ("x_dividends", "inc_netIncome"),
    ),
    # ---------------------------------------------------------------- return: market
    Indicator(
        "ret1y",
        "1-year price return",
        "return",
        "Market",
        "pct",
        True,
        "Price change over the last 12 months (excludes dividends).",
        market=True,
    ),
    Indicator(
        "cagr3y",
        "3-year price CAGR",
        "return",
        "Market",
        "pct",
        True,
        "Annualised price return over 3 years (excludes dividends).",
        market=True,
    ),
    Indicator(
        "cagr5y",
        "5-year price CAGR",
        "return",
        "Market",
        "pct",
        True,
        "Annualised price return over 5 years (excludes dividends).",
        market=True,
    ),
    # ---------------------------------------------------------------- risk: leverage
    Indicator(
        "gearing",
        "Gearing (net debt ÷ equity)",
        "risk",
        "Leverage",
        "pct",
        False,
        "Net debt (total debt − cash & short-term investments) ÷ shareholders' equity. Negative = net cash. "
        "Blank when equity <= 0.",
        _r("bs_netDebt", "bs_totalStockholdersEquity", pos=True),
        ("bs_netDebt", "bs_totalStockholdersEquity"),
    ),
    Indicator(
        "debtToEquity",
        "Debt ÷ equity",
        "risk",
        "Leverage",
        "x",
        False,
        "Total debt ÷ shareholders' equity; blank when equity <= 0.",
        _r("bs_totalDebt", "bs_totalStockholdersEquity", pos=True),
        ("bs_totalDebt", "bs_totalStockholdersEquity"),
    ),
    Indicator(
        "debtToCapital",
        "Debt ÷ capital",
        "risk",
        "Leverage",
        "pct",
        False,
        "Total debt ÷ (total debt + equity) — gearing on a capital basis.",
        _r("bs_totalDebt", "x_debtPlusEquity", pos=True),
        ("bs_totalDebt", "x_debtPlusEquity"),
    ),
    Indicator(
        "debtToAssets",
        "Debt ÷ assets",
        "risk",
        "Leverage",
        "pct",
        False,
        "Total debt ÷ total assets.",
        _r("bs_totalDebt", "bs_totalAssets"),
        ("bs_totalDebt", "bs_totalAssets"),
    ),
    Indicator(
        "netDebtToEbitda",
        "Net debt ÷ EBITDA",
        "risk",
        "Leverage",
        "x",
        False,
        "Years of EBITDA needed to repay net debt; blank when EBITDA <= 0.",
        _r("bs_netDebt", "inc_ebitda", pos=True),
        ("bs_netDebt", "inc_ebitda"),
    ),
    Indicator(
        "interestCoverage",
        "Interest coverage",
        "risk",
        "Leverage",
        "x",
        True,
        "Operating income ÷ interest expense; blank when no interest expense is reported.",
        _r("inc_operatingIncome", "inc_interestExpense", pos=True),
        ("inc_operatingIncome", "inc_interestExpense"),
    ),
    Indicator(
        "equityToAssets",
        "Equity ÷ assets",
        "risk",
        "Leverage",
        "pct",
        True,
        "Shareholders' equity ÷ total assets — balance-sheet capital buffer (key for banks/insurers).",
        _r("bs_totalStockholdersEquity", "bs_totalAssets"),
        ("bs_totalStockholdersEquity", "bs_totalAssets"),
    ),
    Indicator(
        "financialLeverage",
        "Financial leverage (assets ÷ equity)",
        "risk",
        "Leverage",
        "x",
        False,
        "Total assets ÷ shareholders' equity; blank when equity <= 0.",
        _r("bs_totalAssets", "bs_totalStockholdersEquity", pos=True),
        ("bs_totalAssets", "bs_totalStockholdersEquity"),
    ),
    # ---------------------------------------------------------------- risk: liquidity
    Indicator(
        "currentRatio",
        "Current ratio",
        "risk",
        "Liquidity",
        "ratio",
        True,
        "Current assets ÷ current liabilities.",
        _r("bs_totalCurrentAssets", "bs_totalCurrentLiabilities"),
        ("bs_totalCurrentAssets", "bs_totalCurrentLiabilities"),
    ),
    Indicator(
        "quickRatio",
        "Quick ratio",
        "risk",
        "Liquidity",
        "ratio",
        True,
        "(Current assets − inventory) ÷ current liabilities (FMP).",
        _m("r_quickRatio"),
    ),
    Indicator(
        "cashRatio",
        "Cash ratio",
        "risk",
        "Liquidity",
        "ratio",
        True,
        "Cash & equivalents ÷ current liabilities (FMP).",
        _m("r_cashRatio"),
    ),
    # ---------------------------------------------------------------- risk: quality / efficiency
    Indicator(
        "incomeQuality",
        "Earnings quality (OCF ÷ net income)",
        "risk",
        "Earnings quality",
        "ratio",
        True,
        "Operating cash flow ÷ net income; < 1 means earnings are not fully backed by cash. Blank when NI <= 0.",
        _r("cf_operatingCashFlow", "inc_netIncome", pos=True),
        ("cf_operatingCashFlow", "inc_netIncome"),
    ),
    Indicator(
        "cashConversionCycle",
        "Cash conversion cycle",
        "risk",
        "Working capital",
        "days",
        False,
        "Days inventory + days receivables − days payables (FMP).",
        _m("km_cashConversionCycle"),
    ),
    Indicator(
        "inventoryTurnover",
        "Inventory turnover",
        "risk",
        "Working capital",
        "ratio",
        True,
        "Cost of revenue ÷ inventory (FMP).",
        _m("r_inventoryTurnover"),
    ),
    Indicator(
        "capexToRevenue",
        "Capex intensity",
        "risk",
        "Working capital",
        "pct",
        None,
        "Capital expenditure ÷ revenue.",
        _r("x_capex", "inc_revenue"),
        ("x_capex", "inc_revenue"),
    ),
    Indicator(
        "intangiblesToAssets",
        "Goodwill & intangibles ÷ assets",
        "risk",
        "Earnings quality",
        "pct",
        False,
        "Share of assets that are goodwill/intangibles — impairment risk.",
        _r("bs_goodwillAndIntangibleAssets", "bs_totalAssets"),
        ("bs_goodwillAndIntangibleAssets", "bs_totalAssets"),
    ),
    Indicator(
        "sbcToRevenue",
        "Stock-based comp ÷ revenue",
        "risk",
        "Earnings quality",
        "pct",
        False,
        "Stock-based compensation ÷ revenue — dilution cost not in FCF.",
        _r("cf_stockBasedCompensation", "inc_revenue"),
        ("cf_stockBasedCompensation", "inc_revenue"),
    ),
    # ---------------------------------------------------------------- risk: market
    Indicator(
        "volatility",
        "Price volatility (annualised)",
        "risk",
        "Market",
        "pct",
        False,
        f"Annualised std. dev. of daily log returns, last {PRICE_RISK_WINDOW_YEARS} years.",
        market=True,
    ),
    Indicator(
        "maxDrawdown",
        "Max drawdown",
        "risk",
        "Market",
        "pct",
        True,
        f"Largest peak-to-trough price fall over the last {PRICE_HISTORY_YEARS} years.",
        market=True,
    ),
    Indicator(
        "beta",
        "Beta vs market",
        "risk",
        "Market",
        "ratio",
        None,
        f"Slope of weekly returns on the market benchmark, last {PRICE_RISK_WINDOW_YEARS} years "
        "(Newey-West HAC standard errors).",
        market=True,
    ),
    # ---------------------------------------------------------------- valuation
    Indicator(
        "pe",
        "P/E",
        "valuation",
        "Earnings",
        "x",
        False,
        "Market cap ÷ net income (fiscal-year end). Negative = not meaningful.",
        _r("km_marketCap", "inc_netIncome"),
        ("km_marketCap", "inc_netIncome"),
        True,
        True,
    ),
    Indicator(
        "earningsYield",
        "Earnings yield",
        "valuation",
        "Earnings",
        "pct",
        True,
        "Net income ÷ market cap.",
        _r("inc_netIncome", "km_marketCap"),
        ("inc_netIncome", "km_marketCap"),
        True,
    ),
    Indicator(
        "pb",
        "P/B",
        "valuation",
        "Assets",
        "x",
        False,
        "Market cap ÷ shareholders' equity.",
        _r("km_marketCap", "bs_totalStockholdersEquity"),
        ("km_marketCap", "bs_totalStockholdersEquity"),
        True,
        True,
    ),
    Indicator(
        "pTangibleBook",
        "P/Tangible book",
        "valuation",
        "Assets",
        "x",
        False,
        "Market cap ÷ (equity − goodwill & intangibles).",
        _r("km_marketCap", "x_tangibleEquity"),
        ("km_marketCap", "x_tangibleEquity"),
        True,
        True,
    ),
    Indicator(
        "ps",
        "P/S",
        "valuation",
        "Sales",
        "x",
        False,
        "Market cap ÷ revenue.",
        _r("km_marketCap", "inc_revenue"),
        ("km_marketCap", "inc_revenue"),
        True,
        True,
    ),
    Indicator(
        "evToSales",
        "EV/Sales",
        "valuation",
        "Sales",
        "x",
        False,
        "Enterprise value ÷ revenue.",
        _r("km_enterpriseValue", "inc_revenue"),
        ("km_enterpriseValue", "inc_revenue"),
        True,
        True,
    ),
    Indicator(
        "evToEbitda",
        "EV/EBITDA",
        "valuation",
        "Cash flow",
        "x",
        False,
        "Enterprise value ÷ EBITDA.",
        _r("km_enterpriseValue", "inc_ebitda"),
        ("km_enterpriseValue", "inc_ebitda"),
        True,
        True,
    ),
    Indicator(
        "pFcf",
        "P/FCF",
        "valuation",
        "Cash flow",
        "x",
        False,
        "Market cap ÷ free cash flow.",
        _r("km_marketCap", "cf_freeCashFlow"),
        ("km_marketCap", "cf_freeCashFlow"),
        True,
        True,
    ),
    Indicator(
        "fcfYield",
        "FCF yield",
        "valuation",
        "Cash flow",
        "pct",
        True,
        "Free cash flow ÷ market cap.",
        _r("cf_freeCashFlow", "km_marketCap"),
        ("cf_freeCashFlow", "km_marketCap"),
        True,
    ),
    Indicator(
        "pFfo",
        "P/FFO (approx.)",
        "valuation",
        "Cash flow",
        "x",
        False,
        "Market cap ÷ approximate FFO (net income + D&A) — REITs.",
        _r("km_marketCap", "x_ffo"),
        ("km_marketCap", "x_ffo"),
        True,
        True,
    ),
    Indicator(
        "dividendYield",
        "Dividend yield",
        "valuation",
        "Income",
        "pct",
        True,
        "Common dividends paid ÷ market cap.",
        _r("x_dividends", "km_marketCap"),
        ("x_dividends", "km_marketCap"),
        True,
    ),
]

INDICATORS: dict[str, Indicator] = {i.key: i for i in _CATALOG}
PRICE_RAW_COLUMNS = ("km_marketCap", "km_enterpriseValue")

# Current (TTM) valuation from FMP ratios-ttm / key-metrics-ttm, used for the
# "today" snapshot (time series use fiscal-year-end values).
_TTM_FIELDS = {
    "pe": ("ratios", "priceToEarningsRatioTTM"),
    "pb": ("ratios", "priceToBookRatioTTM"),
    "ps": ("ratios", "priceToSalesRatioTTM"),
    "pFcf": ("ratios", "priceToFreeCashFlowRatioTTM"),
    "evToEbitda": ("km", "evToEBITDATTM"),
    "evToSales": ("km", "evToSalesTTM"),
    "dividendYield": ("ratios", "dividendYieldTTM"),
}


def indicators_by_group(group: str) -> list[Indicator]:
    return [i for i in _CATALOG if i.group == group]


def get_profile(sector: str | None, industry: str | None) -> tuple[str, dict[str, Any]]:
    """Industry-specific indicator profile (config.INDUSTRY_INDICATOR_PROFILES)."""
    ind, sec = (industry or "").lower(), (sector or "").lower()
    for key, prof in INDUSTRY_INDICATOR_PROFILES.items():
        if any(m.lower() in ind for m in prof.get("industries", [])):
            return key, dict(prof)
    for key, prof in INDUSTRY_INDICATOR_PROFILES.items():
        if any(m.lower() == sec for m in prof.get("sectors", [])):
            return key, dict(prof)
    return "default", dict(INDUSTRY_INDICATOR_PROFILES["default"])


def profile_defaults(profile: dict[str, Any]) -> dict[str, list[str]]:
    """Default indicator keys per group (a key listed under the wrong group in
    config is placed in its catalog group)."""
    keys = [k for g in GROUPS for k in profile.get(g, []) if k in INDICATORS]
    out: dict[str, list[str]] = {g: [] for g in GROUPS}
    for k in dict.fromkeys(keys):
        out[INDICATORS[k].group].append(k)
    return out


# --------------------------------------------------------------------------- price metrics


def _price_on_or_before(prices: pd.Series, when: pd.Timestamp, tolerance_days: int = 10) -> float | None:
    s = prices.loc[:when]
    if s.empty or (when - s.index[-1]).days > tolerance_days:
        return None
    return float(s.iloc[-1])


def price_metrics(prices: pd.Series, bench: pd.Series | None = None) -> dict[str, Any]:
    """Return and risk statistics from a daily close series (ascending index)."""
    out: dict[str, Any] = {
        k: None for k in ("ret1y", "cagr3y", "cagr5y", "volatility", "maxDrawdown", "beta")
    }
    out.update({"betaP": None, "betaN": None, "betaSE": None})
    p = prices.dropna()
    p = p[p > 0]
    if len(p) < 30:
        return out
    last_date, last = p.index[-1], float(p.iloc[-1])
    for key, yrs in (("ret1y", 1), ("cagr3y", 3), ("cagr5y", 5)):
        start = _price_on_or_before(p, last_date - pd.DateOffset(years=yrs))
        if start:
            out[key] = (last / start) ** (1 / yrs) - 1
    window = p[p.index >= last_date - pd.DateOffset(years=PRICE_RISK_WINDOW_YEARS)]
    logret = np.log(window).diff().dropna()
    if len(logret) > 20:
        out["volatility"] = float(logret.std(ddof=1) * np.sqrt(252))
    out["maxDrawdown"] = float((p / p.cummax() - 1).min())
    if bench is not None and not bench.empty:
        weekly = pd.concat([window.rename("y"), bench.rename("x")], axis=1).dropna().resample("W-FRI").last()
        rets = weekly.apply(np.log).diff().dropna()
        try:
            res = ts.ols_hac(rets["y"], rets[["x"]])
            row = res["coefficients"].loc["x"]
            out.update(
                beta=float(row["coef"]), betaP=float(row["p"]), betaSE=float(row["seHAC"]), betaN=res["n"]
            )
        except ValueError as exc:
            logger.info("Beta not estimated: %s", exc)
    return out


def drawdown_series(prices: pd.Series) -> pd.Series:
    return prices / prices.cummax() - 1


def rolling_volatility(prices: pd.Series, window: int = 252) -> pd.Series:
    return np.log(prices).diff().rolling(window, min_periods=int(window * 0.8)).std() * np.sqrt(252)


# --------------------------------------------------------------------------- company loader


@dataclass
class CompanyData:
    symbol: str
    company: str
    sector: str
    industry: str
    exchange: str
    country: str
    quote_currency: str | None
    reporting_currency: str | None
    currency_mismatch: bool
    market_cap: float | None
    price: float | None
    raw: pd.DataFrame  # fiscalYear-indexed merged statements (prefixed columns) + derived x_ columns
    values: pd.DataFrame  # fiscalYear × indicator key (annual indicators only)
    ttm: dict[str, float | None]  # current valuation (TTM), currency-guarded
    market: dict[str, Any]  # price_metrics output
    prices: pd.Series  # daily close, ascending
    notes: list[str] = field(default_factory=list)

    @property
    def latest_year(self) -> int | None:
        return int(self.values.index.max()) if not self.values.empty else None


def _frame(rows: list[dict[str, Any]], prefix: str) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    fy = df["fiscalYear"] if "fiscalYear" in df else df["date"].astype(str).str[:4]
    df.index = pd.to_numeric(fy, errors="coerce")
    df = df[df.index.notna()]
    df.index = df.index.astype(int)
    df = df[~df.index.duplicated(keep="first")]
    df = df.drop(columns=[c for c in ("symbol", "fiscalYear", "period") if c in df])
    return df.add_prefix(prefix)


def compute_values(raw: pd.DataFrame) -> pd.DataFrame:
    d = derive_raw(raw)
    cols = {i.key: i.compute(d) for i in _CATALOG if i.compute is not None}
    return pd.DataFrame(cols, index=d.index)


def load_prices(symbol: str, client: FMPClient, years: int = PRICE_HISTORY_YEARS) -> pd.Series:
    start = (date.today() - timedelta(days=int(365.25 * years) + 7)).isoformat()
    try:
        rows = client.get_historical_prices(symbol, start=start)
    except FMPError as exc:
        logger.warning("%s: price history unavailable: %s", symbol, exc)
        return pd.Series(dtype=float)
    if not rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame(rows)
    s = pd.Series(pd.to_numeric(df["price"], errors="coerce").to_numpy(), index=pd.to_datetime(df["date"]))
    return s.sort_index().rename(symbol)


def load_company(
    symbol: str,
    client: FMPClient | None = None,
    years: int = INDICATOR_HISTORY_YEARS,
    bench_prices: pd.Series | None = None,
    with_prices: bool = True,
) -> CompanyData:
    """Annual indicator history (up to `years` fiscal years) + current TTM
    valuation + price return/risk for one company. Raises ValueError for an
    unknown symbol and FMPError if the statements cannot be fetched."""
    if years < 1:
        raise ValueError(f"years must be >= 1, got {years}")
    symbol = normalize_symbol(symbol)
    client = client or FMPClient()
    profile = client.get_profile(symbol)
    if not profile:
        raise ValueError(f"No FMP profile for {symbol} — check the ticker")

    inc = _frame(client.get_income_statement(symbol, limit=years), _I)
    if inc.empty:
        raise ValueError(f"No income statements for {symbol}")
    parts = [inc]
    notes: list[str] = []
    for name, fetch, prefix in (
        ("balance sheet", client.get_balance_sheet, _B),
        ("cash flow", client.get_cash_flow, _F),
        ("ratios", client.get_ratios, "r_"),
        ("key metrics", client.get_key_metrics, "km_"),
    ):
        try:
            parts.append(_frame(fetch(symbol, limit=years), prefix))
        except FMPError as exc:  # partial data: keep what we have
            logger.warning("%s: %s unavailable: %s", symbol, name, exc)
            notes.append(f"{name} unavailable — dependent indicators blank")
    raw = pd.concat(parts, axis=1).sort_index()

    quote_ccy = profile.get("currency")
    rep_ccy_by_year = raw.get("inc_reportedCurrency", pd.Series(index=raw.index, dtype=object))
    latest_rep = rep_ccy_by_year.dropna().iloc[-1] if rep_ccy_by_year.notna().any() else None
    if quote_ccy:
        mismatch_rows = rep_ccy_by_year.notna() & (rep_ccy_by_year != quote_ccy)
    else:
        mismatch_rows = pd.Series(False, index=raw.index)
    currency_mismatch = bool(quote_ccy and latest_rep and quote_ccy != latest_rep)
    if mismatch_rows.any():
        logger.warning(
            "%s: quote currency (%s) != reporting currency (%s) in %d fiscal years — nulling price-derived "
            "indicators",
            symbol,
            quote_ccy,
            latest_rep,
            int(mismatch_rows.sum()),
        )
        for col in PRICE_RAW_COLUMNS:
            if col in raw:
                raw.loc[mismatch_rows, col] = np.nan

    raw = derive_raw(raw)
    values = compute_values(raw)
    for i in _CATALOG:
        if i.price_derived and i.key in values:
            values.loc[mismatch_rows, i.key] = np.nan
    values = values.replace([np.inf, -np.inf], np.nan)

    ttm: dict[str, float | None] = {k: None for k in _TTM_FIELDS}
    if not currency_mismatch:
        try:
            src = {"ratios": client.get_ratios_ttm(symbol), "km": client.get_key_metrics_ttm(symbol)}
            for k, (where, fld) in _TTM_FIELDS.items():
                v = src[where].get(fld)
                ttm[k] = float(v) if isinstance(v, int | float) and np.isfinite(v) else None
        except FMPError as exc:
            logger.warning("%s: TTM ratios unavailable: %s", symbol, exc)
            notes.append("TTM valuation unavailable")

    prices = load_prices(symbol, client) if with_prices else pd.Series(dtype=float)
    market = price_metrics(prices, bench_prices) if with_prices else price_metrics(pd.Series(dtype=float))

    return CompanyData(
        symbol=symbol,
        company=str(profile.get("companyName") or symbol),
        sector=str(profile.get("sector") or "—"),
        industry=str(profile.get("industry") or "—"),
        exchange=str(profile.get("exchange") or "—"),
        country=str(profile.get("country") or "—"),
        quote_currency=quote_ccy,
        reporting_currency=latest_rep,
        currency_mismatch=currency_mismatch,
        market_cap=profile.get("marketCap"),
        price=profile.get("price"),
        raw=raw,
        values=values,
        ttm=ttm,
        market=market,
        prices=prices,
        notes=notes,
    )


def trend_findings(c: CompanyData, keys: list[str]) -> list[str]:
    """Plain-language trend statements: latest year vs its own multi-year median."""
    out: list[str] = []
    v = c.values
    if v.empty:
        return out
    rev = _c(c.raw, "inc_revenue").dropna()
    if len(rev) >= 3 and rev.iloc[0] > 0 and rev.iloc[-1] > 0:
        n = int(rev.index[-1] - rev.index[0])
        cagr = (rev.iloc[-1] / rev.iloc[0]) ** (1 / n) - 1
        out.append(
            f"Revenue compounded at **{cagr:+.1%}/yr** over FY{rev.index[0]}–FY{rev.index[-1]} ({n} years)."
        )
    for k in keys:
        ind = INDICATORS.get(k)
        if ind is None or ind.market or k not in v:
            continue
        s = v[k].dropna()
        if len(s) < 4:
            continue
        latest, med = float(s.iloc[-1]), float(s.iloc[:-1].median())
        spread = float(s.iloc[:-1].std(ddof=1)) or 1e-12
        z = (latest - med) / spread
        if abs(z) < 1.0:
            continue
        direction = "above" if z > 0 else "below"
        better = (
            ""
            if ind.higher_is_better is None
            else (" (favourable)" if (z > 0) == ind.higher_is_better else " (unfavourable)")
        )
        fmt = format_value(latest, ind.fmt)
        out.append(
            f"{ind.label} of **{fmt}** in FY{int(s.index[-1])} is {direction} its "
            f"FY{int(s.index[0])}–FY{int(s.index[-2])} median of {format_value(med, ind.fmt)}{better}."
        )
    return out[:8]


def format_value(v: Any, fmt: Fmt) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    if fmt == "pct":
        return f"{v:.1%}"
    if fmt == "x":
        return f"{v:.1f}×"
    if fmt == "days":
        return f"{v:.0f} d"
    return f"{v:.2f}"
