# src/data/store.py
#
# Local columnar data store: Parquet files on disk, queried with DuckDB.
#
#   data/store/                          (gitignored, under data/)
#     fmp/statements/<kind>/<period>/<SYMBOL>.parquet   one file per company
#     fmp/universes/<name>.parquet                       company list snapshots
#     series/<source>/<ID>.parquet                       macro / price series
#
# Design:
# - Upsert, never overwrite: re-syncing merges new rows into the stored
#   history keyed on (date, period), so history accumulates even when an
#   API's look-back window moves forward. Every row carries `fetchedAt`.
# - Writes are atomic (temp file + rename) — an interrupted sync can't leave a
#   half-written Parquet file.
# - Schema: identifier columns stay strings, everything else is coerced to
#   float64, so one company's all-null column can't break a cross-company
#   DuckDB scan (union_by_name).
# - DuckDB reads the files in place. `StoreDB(read_only_sql=True)` loads the
#   tables into memory and then disables DuckDB's file/network access and
#   locks the configuration, so ad-hoc SQL can only read what's loaded.

import re
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from src.config import STORE_DIR
from src.utils.logging import get_logger

logger = get_logger("store")

STATEMENT_KINDS = ("income_statement", "balance_sheet", "cash_flow")
PERIODS = ("annual", "quarter")
_ID_COLUMNS = {
    "date",
    "symbol",
    "reportedCurrency",
    "cik",
    "filingDate",
    "acceptedDate",
    "fiscalYear",
    "period",
    "link",
    "finalLink",
    "fetchedAt",
}
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe(name: str) -> str:
    cleaned = _SAFE.sub("_", name.strip())
    if not cleaned or cleaned in (".", ".."):
        raise ValueError(f"invalid store key {name!r}")
    return cleaned


def _validate_kind(kind: str, period: str) -> None:
    if kind not in STATEMENT_KINDS:
        raise ValueError(f"kind must be one of {STATEMENT_KINDS}, got {kind!r}")
    if period not in PERIODS:
        raise ValueError(f"period must be one of {PERIODS}, got {period!r}")


def statement_path(kind: str, period: str, symbol: str) -> Path:
    _validate_kind(kind, period)
    return STORE_DIR / "fmp" / "statements" / kind / period / f"{_safe(symbol.upper())}.parquet"


def series_path(source: str, series_id: str) -> Path:
    return STORE_DIR / "series" / _safe(source.lower()) / f"{_safe(series_id)}.parquet"


def universe_path(name: str) -> Path:
    return STORE_DIR / "fmp" / "universes" / f"{_safe(name)}.parquet"


def _atomic_write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def _normalise_statements(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for col in df.columns:
        if col in _ID_COLUMNS:
            df[col] = df[col].astype("string")
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    return df


def file_age_hours(path: Path) -> float | None:
    try:
        return (datetime.now().timestamp() - path.stat().st_mtime) / 3600
    except OSError:
        return None


# --------------------------------------------------------------------------- statements


def upsert_statements(kind: str, period: str, symbol: str, rows: list[dict[str, Any]]) -> int:
    """Merge statement rows into the company's Parquet file. Returns rows stored."""
    if not rows:
        return 0
    path = statement_path(kind, period, symbol)
    new = _normalise_statements(rows)
    new["fetchedAt"] = pd.array(
        [datetime.now(timezone.utc).isoformat(timespec="seconds")] * len(new), dtype="string"
    )
    if path.exists():
        old = pd.read_parquet(path)
        merged = pd.concat([old, new], ignore_index=True)
    else:
        merged = new
    merged = (
        merged.sort_values("fetchedAt")
        .drop_duplicates(subset=["date", "period"], keep="last")  # newest fetch wins (restatements)
        .sort_values("date")
        .reset_index(drop=True)
    )
    _atomic_write(merged, path)
    return len(merged)


def load_statements(kind: str, period: str, symbol: str) -> pd.DataFrame:
    path = statement_path(kind, period, symbol)
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def stored_symbols(kind: str, period: str) -> list[str]:
    _validate_kind(kind, period)
    folder = STORE_DIR / "fmp" / "statements" / kind / period
    return sorted(p.stem for p in folder.glob("*.parquet")) if folder.exists() else []


# --------------------------------------------------------------------------- series


def upsert_series(source: str, series_id: str, series: pd.Series, label: str = "") -> None:
    """Merge a dated series into the store (newest value wins per date)."""
    if series.empty:
        return
    path = series_path(source, series_id)
    new = pd.DataFrame({"date": pd.to_datetime(series.index), "value": series.astype(float).to_numpy()})
    if path.exists():
        new = pd.concat([pd.read_parquet(path)[["date", "value"]], new], ignore_index=True)
    new = new.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
    new["source"], new["seriesId"], new["label"] = source, series_id, label
    _atomic_write(new, path)


def load_series(source: str, series_id: str) -> pd.Series | None:
    path = series_path(source, series_id)
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    return pd.Series(df["value"].to_numpy(dtype=float), index=pd.to_datetime(df["date"])).sort_index()


# --------------------------------------------------------------------------- universes


def save_universe(name: str, companies: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(companies)
    keep = [
        c
        for c in (
            "symbol",
            "companyName",
            "sector",
            "industry",
            "country",
            "exchangeShortName",
            "marketCap",
            "beta",
            "price",
            "volume",
        )
        if c in df.columns
    ]
    df = df[keep].copy()
    df["snapshotDate"] = datetime.now(timezone.utc).date().isoformat()
    _atomic_write(df, universe_path(name))
    return df


def load_universe(name: str) -> pd.DataFrame:
    path = universe_path(name)
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def list_universes() -> list[str]:
    folder = STORE_DIR / "fmp" / "universes"
    return sorted(p.stem for p in folder.glob("*.parquet")) if folder.exists() else []


# --------------------------------------------------------------------------- DuckDB


class StoreDB:
    """DuckDB over the Parquet store.

    Tables/views: income_annual, income_quarter, balance_annual, balance_quarter,
    cashflow_annual, cashflow_quarter, companies (union of universe snapshots,
    latest per symbol), series.

    read_only_sql=True materialises everything into memory, then disables
    external access and locks configuration — safe for user-typed SQL.
    """

    _NAMES = {"income_statement": "income", "balance_sheet": "balance", "cash_flow": "cashflow"}

    def __init__(self, read_only_sql: bool = False):
        self.con = duckdb.connect(":memory:")
        self.tables: list[str] = []
        make = "TABLE" if read_only_sql else "VIEW"
        for kind, short in self._NAMES.items():
            for period in PERIODS:
                folder = STORE_DIR / "fmp" / "statements" / kind / period
                if folder.exists() and any(folder.glob("*.parquet")):
                    self._create(
                        make, f"{short}_{period}", f"read_parquet('{folder}/*.parquet', union_by_name=true)"
                    )
        uni = STORE_DIR / "fmp" / "universes"
        if uni.exists() and any(uni.glob("*.parquet")):
            self._create(
                make,
                "companies",
                f"(SELECT * EXCLUDE (rn) FROM (SELECT *, row_number() OVER (PARTITION BY symbol ORDER BY snapshotDate DESC) rn "
                f"FROM read_parquet('{uni}/*.parquet', union_by_name=true)) WHERE rn = 1)",
            )
        ser = STORE_DIR / "series"
        if ser.exists() and any(ser.rglob("*.parquet")):
            self._create(make, "series", f"read_parquet('{ser}/*/*.parquet', union_by_name=true)")
        if read_only_sql:
            self.con.execute("SET enable_external_access = false")
            self.con.execute("SET lock_configuration = true")

    def _create(self, make: str, name: str, source: str) -> None:
        self.con.execute(f"CREATE {make} {name} AS SELECT * FROM {source}")
        self.tables.append(name)

    def query(self, sql: str, params: Iterable[Any] | None = None) -> pd.DataFrame:
        return self.con.execute(sql, list(params) if params is not None else None).df()

    def query_user(self, sql: str, max_rows: int = 5000) -> pd.DataFrame:
        """Run user-typed SQL: single SELECT/WITH statement only, row-capped."""
        stripped = sql.strip().rstrip(";").strip()
        if not re.match(r"(?is)^(select|with)\b", stripped) or ";" in stripped:
            raise ValueError("Only a single SELECT (or WITH … SELECT) statement is allowed")
        # Cap via the relation API, not string wrapping: a trailing "--" comment in the
        # user's SQL would otherwise swallow an appended ") LIMIT n".
        # Check the parsed statement type BEFORE executing: con.sql() runs DML immediately
        # ("WITH t AS (SELECT 1) DELETE ..." passes the prefix regex).
        try:
            statements = self.con.extract_statements(stripped)
        except duckdb.Error as exc:
            raise ValueError(f"SQL error: {exc}") from exc
        if len(statements) != 1 or statements[0].type != duckdb.StatementType.SELECT:
            raise ValueError("Only a single SELECT (or WITH … SELECT) statement is allowed")
        rel = self.con.sql(stripped)
        if rel is None:
            raise ValueError("Query returned no result set")
        return rel.limit(int(max_rows)).df()

    def describe(self) -> pd.DataFrame:
        rows = []
        for t in self.tables:
            n = self.con.execute(f"SELECT count(*) FROM {t}").fetchone()
            cols = self.con.execute(f"SELECT count(*) FROM (DESCRIBE {t})").fetchone()
            rows.append({"table": t, "rows": n[0] if n else 0, "columns": cols[0] if cols else 0})
        return pd.DataFrame(rows)

    def close(self) -> None:
        self.con.close()
