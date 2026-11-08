# src/backtest/data.py
#
# Point-in-time inputs for the backtester, built once per run from the local store:
#   - month-end total-return prices (dividend-adjusted close) and split-adjusted close
#   - the 3-month T-bill as the monthly risk-free rate (known at the START of each month)
#   - annual fundamentals stamped with the date they became PUBLIC:
#       availableFrom = max(acceptedDate or filingDate, fiscal year end + MIN_REPORT_LAG) + LAG
#     A rebalance on date t may only use rows with availableFrom <= t — no look-ahead.
# Data stays read-only; prices come from `python -m src.data.sync --prices`.

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from pandas.tseries.holiday import GoodFriday, USMemorialDay

from src.config import (
    BACKTEST_FILING_LAG_DAYS,
    BACKTEST_LIQUIDITY_WINDOW,
    BACKTEST_MARKET_ETFS,
    BACKTEST_MIN_CAP,
    BACKTEST_MIN_REPORT_LAG_DAYS,
    BACKTEST_RISK_FREE_SERIES,
    BACKTEST_SECTOR_ETFS,
    FISCAL_YEAR_END_GRACE_DAYS,
    STORE_DIR,
)
from src.data import store
from src.data.store import StoreDB
from src.utils.logging import get_logger

logger = get_logger("backtest.data")

_PIT_SQL = """
WITH raw AS (
    SELECT symbol, CAST(date AS DATE) AS periodEnd, reportedCurrency AS currency,
           coalesce(TRY_CAST(acceptedDate AS DATE), TRY_CAST(filingDate AS DATE)) AS filed,
           revenue, grossProfit, operatingIncome, netIncome, weightedAverageShsOutDil AS shares,
           interestIncome, interestExpense, netInterestIncome
    FROM income_annual WHERE period = 'FY'
), r AS (
    -- fullYear is False only for a part-year period: one ending < 330 days after the previous
    -- FILED period (ZWS Apr-Dec 2020, 275 days). A longer gap means a missing predecessor in FMP,
    -- not a short year (APC FY2022 after 1,461 days is a real 12-month year).
    SELECT *, coalesce(
               periodEnd - lag(periodEnd) OVER (PARTITION BY symbol ORDER BY periodEnd) >= 330,
               TRUE) AS fullYear
    FROM raw
), i AS (
    -- One row per company fiscal year (same rule as fundamentals_panel): 52/53-week years ending
    -- in early January belong to the prior year. Prefer the full-year row over a stub in the same
    -- year (ZWS: keep FY Mar-2020, not the 9-month Dec-2020 stub), then one with revenue (LHX
    -- FY2019 revenue 0), then the later period.
    SELECT * FROM r
    QUALIFY row_number() OVER (
        PARTITION BY symbol, year(periodEnd - INTERVAL {grace} DAY)
        ORDER BY fullYear DESC, (coalesce(revenue, 0) > 0) DESC, periodEnd DESC
    ) = 1
), b AS (
    SELECT symbol, CAST(date AS DATE) AS periodEnd,
           totalAssets, totalStockholdersEquity AS equity, totalDebt,
           totalCurrentAssets, totalCurrentLiabilities, netReceivables, accountPayables
    FROM balance_annual WHERE period = 'FY'
), c AS (
    SELECT symbol, CAST(date AS DATE) AS periodEnd, freeCashFlow
    FROM cashflow_annual WHERE period = 'FY'
), j AS (
    SELECT i.*, b.totalAssets, b.equity, b.totalDebt, b.totalCurrentAssets, b.totalCurrentLiabilities,
           b.netReceivables, b.accountPayables, c.freeCashFlow,
           lag(i.revenue) OVER w AS lagRevenue, lag(i.netIncome) OVER w AS lagNetIncome, lag(i.periodEnd) OVER w AS lagPeriodEnd,
           lag(i.fullYear) OVER w AS lagFullYear
    FROM i LEFT JOIN b USING (symbol, periodEnd) LEFT JOIN c USING (symbol, periodEnd)
    WINDOW w AS (PARTITION BY i.symbol ORDER BY i.periodEnd)
)
SELECT symbol, periodEnd, currency, fullYear,
       greatest(coalesce(filed, periodEnd + INTERVAL {min_lag} DAY), periodEnd + INTERVAL {min_lag} DAY)
           + INTERVAL {lag} DAY AS availableFrom,
       revenue, grossProfit, operatingIncome, netIncome, shares,
       -- growth only between two full fiscal years 330-400 days apart (never against a stub)
       CASE WHEN fullYear AND lagFullYear AND periodEnd - lagPeriodEnd BETWEEN 330 AND 400
            THEN lagRevenue END AS prevRevenue,
       CASE WHEN fullYear AND lagFullYear AND periodEnd - lagPeriodEnd BETWEEN 330 AND 400
            THEN lagNetIncome END AS prevNetIncome,
       interestIncome, interestExpense, netInterestIncome,
       totalAssets, equity, totalDebt, totalCurrentAssets, totalCurrentLiabilities,
       netReceivables, accountPayables, freeCashFlow
FROM j
"""


@dataclass
class BacktestData:
    adj: pd.DataFrame  # month-end dividend-adjusted close (total-return index), date × symbol
    close: pd.DataFrame  # month-end split-adjusted close (for point-in-time market cap)
    returns: pd.DataFrame  # monthly total return for the month ENDING at each index date
    rf: pd.Series  # monthly risk-free return for the month ending at each index date
    pit: pd.DataFrame  # point-in-time annual fundamentals (see _PIT_SQL)
    sectors: pd.Series  # symbol -> FMP sector (today's classification; caveat)
    etfs: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    # month-end label -> last actual trading date in that month (the close a rebalance trades at)
    trade_dates: pd.Series = field(default_factory=lambda: pd.Series(dtype="datetime64[ns]"))
    # memo for derived per-date values (e.g. the universe at t), shared by every strategy in a run
    memo: dict[Any, Any] = field(default_factory=dict, repr=False)
    # 50- and 200-trading-day moving averages of the daily close, sampled at each month's last
    # session (the scorecard's trend test: price vs its 50D / 200D averages)
    ma50: pd.DataFrame = field(default_factory=pd.DataFrame)
    ma200: pd.DataFrame = field(default_factory=pd.DataFrame)
    # median daily dollar volume over the trailing liquidity window, at each month's last session
    dollar_volume: pd.DataFrame = field(default_factory=pd.DataFrame)
    industries: pd.Series = field(default_factory=lambda: pd.Series(dtype=object))  # today's FMP industry
    names: pd.Series = field(default_factory=lambda: pd.Series(dtype=object))  # company name

    def fundamentals_asof(self, t: pd.Timestamp) -> pd.DataFrame:
        """Latest annual fundamentals per symbol that were PUBLIC at the close the rebalance
        labelled `t` trades at — the month's last TRADING day, not the calendar month-end (a
        filing accepted after Friday's close must not be traded at that close)."""
        cutoff = self.trade_dates.get(t, t)
        avail = self.pit[self.pit["availableFrom"] <= cutoff]
        if avail.empty:
            return avail.set_index("symbol")
        return avail.sort_values("periodEnd").groupby("symbol").tail(1).set_index("symbol")


def _month_end(df: pd.DataFrame) -> pd.DataFrame:
    return df.resample("ME").last()


def load_backtest_data(symbols: list[str] | None = None, db: StoreDB | None = None) -> BacktestData:
    """Assemble prices, risk-free rate and point-in-time fundamentals from the store."""
    db = db or StoreDB()
    if "income_annual" not in db.tables:
        raise ValueError("No statements in the store — run `python -m src.data.sync` first")
    etfs = [*BACKTEST_MARKET_ETFS, *BACKTEST_SECTOR_ETFS]
    have = set(store.stored_price_symbols())
    if not have:
        raise ValueError("No prices in the store — run `python -m src.data.sync --prices --no-statements`")
    syms = sorted((set(symbols) if symbols else have) | (set(etfs) & have))
    adj_cols, close_cols, ma50_cols, ma200_cols, dv_cols = {}, {}, {}, {}, {}
    for s in syms:
        p = store.load_prices(s)
        if p.empty:
            continue
        p = p.set_index("date").sort_index()
        p.index = pd.DatetimeIndex(p.index)
        adj_cols[s], close_cols[s] = p["adjClose"], p["close"]
        ma50_cols[s] = p["close"].rolling(50, min_periods=50).mean()
        ma200_cols[s] = p["close"].rolling(200, min_periods=200).mean()
        if "volume" in p:
            w = int(BACKTEST_LIQUIDITY_WINDOW)
            dv_cols[s] = (p["close"] * p["volume"]).rolling(w, min_periods=w // 2).median()
    adj = _month_end(pd.DataFrame(adj_cols))
    close = _month_end(pd.DataFrame(close_cols))
    # Month-end resampling carries the last price of a symbol with no trades that month; a
    # symbol that stopped trading would otherwise show a flat price forever.
    for s, ser in adj_cols.items():
        last = ser.last_valid_index()
        if last is None:
            continue
        cutoff = pd.Timestamp(str(last)) + pd.offsets.MonthEnd(0)
        adj[s] = adj[s].where(pd.Series(adj.index <= cutoff, index=adj.index))
    # Drop months that the PRICE DATA hasn't completed. Keyed to the benchmark's last trading date,
    # not the wall clock: prices synced mid-October and a backtest run in November would otherwise
    # present a half month as October (SPY +1.20% vs a true -0.91%).
    ref = adj_cols.get("SPY")
    ref_dates = (
        pd.DatetimeIndex(ref.dropna().index)
        if ref is not None
        else pd.DatetimeIndex(next(iter(adj_cols.values())).index)
    )
    last_px = ref_dates.max()
    # month-end label -> the month's last trading date (index labels match the "ME" resample)
    trade = pd.Series(ref_dates, index=ref_dates).groupby(ref_dates + pd.offsets.MonthEnd(0)).max()
    # Every month before the last price's month is complete; that month counts only if the last
    # price IS its final session (data synced after the close of the month's last trading day).
    month_complete = [d for d in pd.DatetimeIndex(adj.index) if d < last_px or _is_last_session(last_px, d)]
    complete = pd.DatetimeIndex(month_complete)
    adj, close = adj.loc[complete], close.loc[complete]
    ma50 = _month_end(pd.DataFrame(ma50_cols)).reindex(complete)
    ma200 = _month_end(pd.DataFrame(ma200_cols)).reindex(complete)
    dollar_volume = _month_end(pd.DataFrame(dv_cols)).reindex(complete)
    returns = adj.pct_change(fill_method=None)

    rf = _risk_free(pd.DatetimeIndex(adj.index))
    pit = db.query(
        _PIT_SQL.format(
            min_lag=int(BACKTEST_MIN_REPORT_LAG_DAYS),
            lag=int(BACKTEST_FILING_LAG_DAYS),
            grace=int(FISCAL_YEAR_END_GRACE_DAYS),
        )
    )
    pit["availableFrom"] = pd.to_datetime(pit["availableFrom"])
    pit["periodEnd"] = pd.to_datetime(pit["periodEnd"])
    comp = (
        db.query("SELECT symbol, companyName, sector, industry FROM companies").drop_duplicates("symbol")
        if "companies" in db.tables
        else pd.DataFrame(columns=["symbol", "companyName", "sector", "industry"])
    ).set_index("symbol")
    sectors = comp["sector"]
    notes = [
        f"{len(adj.columns)} symbols with prices ({len(set(etfs) & set(adj.columns))} ETFs), "
        f"{adj.index.min():%Y-%m} to {adj.index.max():%Y-%m}; {pit['symbol'].nunique()} companies with "
        "point-in-time annual fundamentals."
    ]
    notes.append(_universe_source_caveat(db, [c for c in adj.columns if c not in set(etfs)]))
    trade_dates = trade.reindex(pd.DatetimeIndex(adj.index))
    return BacktestData(
        adj,
        close,
        returns,
        rf,
        pit,
        sectors,
        [e for e in etfs if e in adj.columns],
        notes,
        trade_dates,
        ma50=ma50,
        ma200=ma200,
        dollar_volume=dollar_volume,
        industries=comp["industry"],
        names=comp["companyName"],
    )


def _universe_source_caveat(db: StoreDB, symbols: list[str]) -> str:
    """The candidate list is a snapshot of companies listed TODAY, so membership at past dates is
    point-in-time only within it: a company >= the cap floor years ago that is now below the
    snapshot's floor, delisted, acquired or bankrupt is missing (look-ahead in selection)."""
    floor, snap = float("nan"), "unknown date"
    try:
        if "companies" in db.tables and symbols:
            caps = db.query("SELECT symbol, marketCap FROM companies").set_index("symbol")["marketCap"]
            floor = float(pd.to_numeric(caps.reindex(symbols), errors="coerce").min())
        files = list(STORE_DIR.glob("fmp/universes/*.parquet"))
        if files:
            snap = pd.Timestamp(max(f.stat().st_mtime for f in files), unit="s").strftime("%Y-%m-%d")
    except Exception as exc:  # a caveat must never break the data load
        logger.warning("Universe source caveat incomplete (%s)", exc)
    floor_txt = f"${floor / 1e9:.1f}bn" if pd.notna(floor) else "an unknown floor"
    return (
        f"Universe source: {len(symbols)} companies listed today (screener snapshot {snap}, smallest current cap "
        f"{floor_txt}); point-in-time cap ≥ ${BACKTEST_MIN_CAP / 1e9:.0f}bn is applied within that list. Companies "
        "that met the floor in the past but are now smaller than the snapshot floor, delisted, acquired or bankrupt "
        "are missing — a look-ahead in selection that flatters stock strategies."
    )


def _is_last_session(last: pd.Timestamp, month_end: pd.Timestamp) -> bool:
    """True if `last` is on/after the month's final NYSE session: the last weekday that isn't a
    holiday which can fall at a month-end (Memorial Day on 31 May, Good Friday on 31 Mar / 30 Apr;
    New Year's Day and Christmas aren't month-ends). A store synced one session short of the
    month-end (e.g. Thu 29 Oct when Fri 30 Oct traded) is NOT complete."""
    start = month_end.replace(day=1)
    closed = set(USMemorialDay.dates(start, month_end)) | set(GoodFriday.dates(start, month_end))
    sessions = [d for d in pd.bdate_range(start, month_end) if d not in closed]
    return bool(sessions) and last >= sessions[-1]


def _risk_free(index: pd.DatetimeIndex) -> pd.Series:
    """Monthly risk-free return for each month ending at `index`: the T-bill yield known at
    the START of the month (previous month's average), / 12. Falls back to 0 with a warning."""
    try:
        from src.data.series_loader import load_series

        tb = load_series(BACKTEST_RISK_FREE_SERIES).series  # % p.a., month-start dated
        tb.index = tb.index + pd.offsets.MonthEnd(0)  # value for month m, stamped at its end
        rf = (tb.shift(1) / 100 / 12).reindex(index).ffill()
        return rf.fillna(0.0)
    except Exception as exc:  # report-path rule: degrade, never raise
        logger.warning("Risk-free series unavailable (%s) — using 0", exc)
        return pd.Series(0.0, index=index)
