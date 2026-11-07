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

import pandas as pd

from src.config import (
    BACKTEST_FILING_LAG_DAYS,
    BACKTEST_MARKET_ETFS,
    BACKTEST_MIN_REPORT_LAG_DAYS,
    BACKTEST_RISK_FREE_SERIES,
    BACKTEST_SECTOR_ETFS,
)
from src.data import store
from src.data.store import StoreDB
from src.utils.logging import get_logger

logger = get_logger("backtest.data")

_PIT_SQL = """
WITH i AS (
    SELECT symbol, CAST(date AS DATE) AS periodEnd,
           coalesce(TRY_CAST(acceptedDate AS DATE), TRY_CAST(filingDate AS DATE)) AS filed,
           revenue, grossProfit, operatingIncome, netIncome, weightedAverageShsOutDil AS shares
    FROM income_annual WHERE period = 'FY'
), b AS (
    SELECT symbol, CAST(date AS DATE) AS periodEnd,
           totalAssets, totalStockholdersEquity AS equity, totalDebt
    FROM balance_annual WHERE period = 'FY'
), c AS (
    SELECT symbol, CAST(date AS DATE) AS periodEnd, freeCashFlow
    FROM cashflow_annual WHERE period = 'FY'
)
SELECT i.symbol, i.periodEnd,
       greatest(coalesce(i.filed, i.periodEnd + INTERVAL {min_lag} DAY), i.periodEnd + INTERVAL {min_lag} DAY)
           + INTERVAL {lag} DAY AS availableFrom,
       i.revenue, i.grossProfit, i.operatingIncome, i.netIncome, i.shares,
       lag(i.revenue) OVER (PARTITION BY i.symbol ORDER BY i.periodEnd) AS prevRevenue,
       b.totalAssets, b.equity, b.totalDebt, c.freeCashFlow
FROM i LEFT JOIN b USING (symbol, periodEnd) LEFT JOIN c USING (symbol, periodEnd)
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

    def fundamentals_asof(self, t: pd.Timestamp) -> pd.DataFrame:
        """Latest annual fundamentals per symbol that were PUBLIC on date t."""
        avail = self.pit[self.pit["availableFrom"] <= t]
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
    adj_cols, close_cols = {}, {}
    for s in syms:
        p = store.load_prices(s)
        if p.empty:
            continue
        p = p.set_index("date").sort_index()
        adj_cols[s], close_cols[s] = p["adjClose"], p["close"]
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
    # Drop a trailing month that hasn't ended yet: its "month-end" price is just the latest
    # close, so its return would be a partial month presented as a full one.
    today = pd.Timestamp.today().normalize()
    complete = pd.DatetimeIndex([d for d in pd.DatetimeIndex(adj.index) if d <= today])
    adj, close = adj.loc[complete], close.loc[complete]
    returns = adj.pct_change(fill_method=None)

    rf = _risk_free(pd.DatetimeIndex(adj.index))
    pit = db.query(
        _PIT_SQL.format(min_lag=int(BACKTEST_MIN_REPORT_LAG_DAYS), lag=int(BACKTEST_FILING_LAG_DAYS))
    )
    pit["availableFrom"] = pd.to_datetime(pit["availableFrom"])
    pit["periodEnd"] = pd.to_datetime(pit["periodEnd"])
    sectors = (
        db.query("SELECT symbol, sector FROM companies").set_index("symbol")["sector"]
        if "companies" in db.tables
        else pd.Series(dtype=object)
    )
    notes = [
        f"{len(adj.columns)} symbols with prices ({len(set(etfs) & set(adj.columns))} ETFs), "
        f"{adj.index.min():%Y-%m} to {adj.index.max():%Y-%m}; {pit['symbol'].nunique()} companies with "
        "point-in-time annual fundamentals."
    ]
    return BacktestData(adj, close, returns, rf, pit, sectors, [e for e in etfs if e in adj.columns], notes)


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
