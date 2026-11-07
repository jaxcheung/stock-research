# src/data/sync.py
#
# Bulk download of FMP financial statements into the Parquet store.
#
#   python -m src.data.sync                         # US-listed >= $10B, 10y income statements
#   python -m src.data.sync --min-cap 2e9 --kinds income_statement balance_sheet
#   python -m src.data.sync --symbols NIO XPEV LI BYDDY --universe china_autos
#
# Resilient by design: each company is fetched and stored independently (one
# 402/404 never stops the batch), files refreshed within max_age are skipped
# (cheap re-runs / resume after interruption), and calls are throttled under
# the Starter-tier rate limit by FMPClient.

import argparse
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.config import (
    BACKTEST_MARKET_ETFS,
    BACKTEST_PRICE_START,
    BACKTEST_SECTOR_ETFS,
    SYNC_DEFAULT_YEARS,
    SYNC_UNIVERSE_MIN_CAP,
)
from src.config import STORE_STATEMENT_MAX_AGE_HOURS as MAX_AGE
from src.data import store
from src.data.fmp_client import FMPClient, FMPError, normalize_symbol
from src.utils.logging import get_logger

logger = get_logger("sync")

_FETCH = {
    "income_statement": "get_income_statement",
    "balance_sheet": "get_balance_sheet",
    "cash_flow": "get_cash_flow",
}
_MAJOR_EXCHANGES = ("NASDAQ", "NYSE", "AMEX")
# Exchange-listed debt / preferred lines (baby bonds, depositary preferreds)
# appear in the screener like stocks; they are not operating companies.
_NON_EQUITY = re.compile(
    r"\b(notes?|debentures?|preferred|pfd|depositary shares|fixed[- ]to[- ]floating|due \d{4})\b"
    r"|\d+(\.\d+)?\s*%|\bfinance co\b",
    re.IGNORECASE,
)


def is_operating_equity(row: dict[str, Any]) -> bool:
    return not _NON_EQUITY.search(row.get("companyName") or "")


@dataclass
class SyncReport:
    universe: str
    requested: int = 0
    downloaded: list[str] = field(default_factory=list)
    skipped_fresh: list[str] = field(default_factory=list)
    empty: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    rows_stored: int = 0

    def summary(self) -> str:
        return (
            f"{self.universe}: {self.requested} requested · {len(self.downloaded)} downloaded · "
            f"{len(self.skipped_fresh)} already fresh · {len(self.empty)} no data · {len(self.failed)} failed · "
            f"{self.rows_stored:,} rows in store"
        )


def build_universe(
    name: str = "us_large",
    min_cap: float = SYNC_UNIVERSE_MIN_CAP,
    country: str | None = "US",
    client: FMPClient | None = None,
) -> list[dict[str, Any]]:
    """Screen US-listed operating companies (no ETFs/funds), de-duplicate share
    classes (GOOG/GOOGL), snapshot to the store, return rows by market cap."""
    if min_cap <= 0:
        raise ValueError(f"min_cap must be > 0, got {min_cap}")
    client = client or FMPClient()
    rows = client.screen_companies(limit=5000, country=country, marketCapMoreThan=int(min_cap))
    rows = [
        r
        for r in rows
        if r.get("exchangeShortName") in _MAJOR_EXCHANGES
        and not r.get("isEtf")
        and not r.get("isFund")
        and r.get("symbol")
        and "." not in r["symbol"]
        and is_operating_equity(r)
    ]
    by_name: dict[str, dict[str, Any]] = {}
    for r in rows:
        key = r.get("companyName") or r["symbol"]
        if key not in by_name or (r.get("volume") or 0) > (by_name[key].get("volume") or 0):
            by_name[key] = r
    out = sorted(by_name.values(), key=lambda r: r.get("marketCap") or 0, reverse=True)
    store.save_universe(name, out)
    logger.info(
        "Universe %s: %d companies (min cap $%.0fB, country=%s)", name, len(out), min_cap / 1e9, country
    )
    return out


def sync_statements(
    symbols: list[str],
    universe: str,
    kinds: tuple[str, ...] = ("income_statement",),
    period: str = "annual",
    years: int = SYNC_DEFAULT_YEARS,
    max_age_hours: float = MAX_AGE,
    client: FMPClient | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> SyncReport:
    """Download statements for `symbols` into the store. `progress(i, n, symbol)`
    is called before each company (for a UI progress bar)."""
    if not symbols:
        raise ValueError("symbols must be non-empty")
    if years < 1:
        raise ValueError(f"years must be >= 1, got {years}")
    for k in kinds:
        if k not in _FETCH:
            raise ValueError(f"unknown statement kind {k!r}; expected {tuple(_FETCH)}")
    client = client or FMPClient()
    limit = years if period == "annual" else years * 4
    rep = SyncReport(universe=universe, requested=len(symbols))
    syms = list(dict.fromkeys(normalize_symbol(s) for s in symbols))
    for i, sym in enumerate(syms):
        if progress:
            progress(i, len(syms), sym)
        try:
            fresh = all(
                (age := store.file_age_hours(store.statement_path(k, period, sym))) is not None
                and age < max_age_hours
                for k in kinds
            )
            if fresh:
                rep.skipped_fresh.append(sym)
                continue
            got_any = False
            for k in kinds:
                rows = getattr(client, _FETCH[k])(sym, period=period, limit=limit)
                if rows:
                    rep.rows_stored += store.upsert_statements(k, period, sym, rows)
                    got_any = True
            (rep.downloaded if got_any else rep.empty).append(sym)
        except (FMPError, ValueError, OSError) as exc:
            rep.failed[sym] = str(exc)
            logger.warning("Sync %s failed: %s", sym, exc)
        if (i + 1) % 50 == 0:
            logger.info("Sync progress %d/%d (%d failed)", i + 1, len(syms), len(rep.failed))
    if progress:
        progress(len(syms), len(syms), "")
    logger.info(rep.summary())
    return rep


def sync_prices(
    symbols: list[str],
    start: str,
    max_age_hours: float = MAX_AGE,
    client: FMPClient | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> SyncReport:
    """Daily split-adjusted close AND dividend-adjusted close per symbol into the store
    (two calls each). Total-return backtests use adjClose; point-in-time market cap uses
    close x shares from the latest filed statement."""
    if not symbols:
        raise ValueError("symbols must be non-empty")
    client = client or FMPClient()
    rep = SyncReport(universe="prices", requested=len(symbols))
    syms = list(dict.fromkeys(normalize_symbol(s) for s in symbols))
    for i, sym in enumerate(syms):
        if progress:
            progress(i, len(syms), sym)
        age = store.file_age_hours(store.price_path(sym))
        if age is not None and age < max_age_hours:
            rep.skipped_fresh.append(sym)
            continue
        try:
            full = client.get_price_history_full(sym, start=start)
            adj = client.get_price_history_adjusted(sym, start=start)
            if not full or not adj:
                rep.empty.append(sym)
                continue
            f = pd.DataFrame(full)[["date", "close", "volume"]]
            a = pd.DataFrame(adj)[["date", "adjClose"]]
            rep.rows_stored += store.upsert_prices(sym, f.merge(a, on="date", how="inner"))
            rep.downloaded.append(sym)
        except (FMPError, ValueError, OSError, KeyError) as exc:
            rep.failed[sym] = str(exc)
            logger.warning("Price sync %s failed: %s", sym, exc)
        if (i + 1) % 100 == 0:
            logger.info("Price sync progress %d/%d (%d failed)", i + 1, len(syms), len(rep.failed))
    logger.info(rep.summary())
    return rep


def main() -> None:
    ap = argparse.ArgumentParser(description="Download FMP statements into the local Parquet store")
    ap.add_argument("--universe", default="us_large", help="name for the company-list snapshot")
    ap.add_argument(
        "--min-cap", type=float, default=SYNC_UNIVERSE_MIN_CAP, help="screener market-cap floor (USD)"
    )
    ap.add_argument("--country", default="US", help="screener country filter ('' for any)")
    ap.add_argument("--symbols", nargs="*", help="explicit tickers instead of a screened universe")
    ap.add_argument("--kinds", nargs="+", default=["income_statement"], choices=list(_FETCH))
    ap.add_argument("--period", default="annual", choices=["annual", "quarter"])
    ap.add_argument("--years", type=int, default=SYNC_DEFAULT_YEARS)
    ap.add_argument("--max-age-hours", type=float, default=MAX_AGE)
    ap.add_argument(
        "--prices",
        action="store_true",
        help="also sync daily close + dividend-adjusted close for the symbols and the backtest ETFs",
    )
    ap.add_argument("--no-statements", action="store_true", help="skip statements (prices only)")
    ap.add_argument("--price-start", default=BACKTEST_PRICE_START)
    args = ap.parse_args()

    client = FMPClient()
    if args.symbols:
        companies = []
        for s in args.symbols:
            try:
                p = client.get_profile(s)
            except FMPError:
                p = {}
            companies.append(
                {
                    "symbol": normalize_symbol(s),
                    **{k: p.get(k) for k in ("companyName", "sector", "industry", "country", "marketCap")},
                }
            )
        store.save_universe(args.universe, companies)
        symbols = [str(c["symbol"]) for c in companies]
    else:
        symbols = [
            r["symbol"] for r in build_universe(args.universe, args.min_cap, args.country or None, client)
        ]
    if not args.no_statements:
        sync_statements(
            symbols, args.universe, tuple(args.kinds), args.period, args.years, args.max_age_hours, client
        )
    if args.prices:
        etfs = [*BACKTEST_MARKET_ETFS, *BACKTEST_SECTOR_ETFS]
        sync_prices([*etfs, *symbols], args.price_start, args.max_age_hours, client)


if __name__ == "__main__":
    main()
