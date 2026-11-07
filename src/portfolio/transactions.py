# src/portfolio/transactions.py

import math
from datetime import date
from pathlib import Path
from typing import Any

from src.config import PORTFOLIO_QTY_TOLERANCE
from src.data.fmp_client import normalize_symbol
from src.portfolio.db import get_connection
from src.utils.logging import get_logger

logger = get_logger("portfolio.transactions")


def _validate_date(transaction_date: str) -> str:
    try:
        parsed = date.fromisoformat(transaction_date)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"transaction_date must be ISO format YYYY-MM-DD, got {transaction_date!r}") from exc
    if parsed > date.today():
        raise ValueError(f"transaction_date {transaction_date} is in the future")
    return parsed.isoformat()


def _validate_number(name: str, value: float, *, allow_zero: bool) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number, got {value!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    if value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{name} must be {'>= 0' if allow_zero else '> 0'}, got {value}")
    return value


def get_held_quantity(symbol: str, db_path: Path | None = None) -> float:
    """Net shares currently held (sum of BUYs minus sum of SELLs)."""
    symbol = normalize_symbol(symbol)
    with get_connection(db_path) as conn:
        row = conn.execute(
            """
            SELECT
                COALESCE(SUM(CASE WHEN transaction_type = 'BUY' THEN quantity ELSE 0 END), 0)
                - COALESCE(SUM(CASE WHEN transaction_type = 'SELL' THEN quantity ELSE 0 END), 0) AS held
            FROM transactions WHERE symbol = ?
            """,
            (symbol,),
        ).fetchone()
    return float(row["held"])


def record_buy(
    symbol: str,
    quantity: float,
    price: float,
    transaction_date: str,
    fees: float = 0.0,
    notes: str | None = None,
    db_path: Path | None = None,
) -> int:
    return _record(symbol, "BUY", quantity, price, transaction_date, fees, notes, db_path)


def record_sell(
    symbol: str,
    quantity: float,
    price: float,
    transaction_date: str,
    fees: float = 0.0,
    notes: str | None = None,
    db_path: Path | None = None,
) -> int:
    symbol = normalize_symbol(symbol)
    quantity = _validate_number("quantity", quantity, allow_zero=False)
    transaction_date = _validate_date(transaction_date)
    _check_sell_against_timeline(symbol, quantity, transaction_date, db_path)
    return _record(symbol, "SELL", quantity, price, transaction_date, fees, notes, db_path)


def _unmatched_sells(timeline: list[tuple[str, float, str, float]]) -> tuple[set[float], dict[float, float]]:
    """(ids of SELLs that FIFO cannot fully match, shares held just before each SELL),
    replayed exactly as pnl.compute_fifo does: an unmatched remainder is dropped (held
    floors at 0), not carried negative."""
    held = 0.0
    bad: set[float] = set()
    held_before: dict[float, float] = {}
    for _, txn_id, txn_type, qty in sorted(timeline, key=lambda t: (t[0], t[1])):
        if txn_type == "BUY":
            held += qty
            continue
        held_before[txn_id] = held
        if qty > held + PORTFOLIO_QTY_TOLERANCE:
            bad.add(txn_id)
        held = max(held - qty, 0.0)
    return bad, held_before


def _check_sell_against_timeline(
    symbol: str, quantity: float, transaction_date: str, db_path: Path | None
) -> None:
    """Reject a SELL unless it, and every SELL after it, stays fully FIFO-matchable.

    FIFO replays transactions in (date, id) order, so the check must too: a backdated
    SELL can exceed what was held on its date even when the all-time net is enough, and
    it can also starve a later, already-recorded SELL. The new row will get the highest
    id, so it sorts after any same-day BUYs. SELLs that were already unmatchable before
    this one (legacy rows from before this validation existed) are not held against it.
    """
    existing: list[tuple[str, float, str, float]] = [
        (t["transaction_date"], float(t["id"]), t["transaction_type"], float(t["quantity"]))
        for t in list_transactions(symbol, db_path=db_path)
    ]
    legacy_bad, _ = _unmatched_sells(existing)
    if legacy_bad:
        logger.warning("%s has %d legacy SELL(s) FIFO cannot fully match", symbol, len(legacy_bad))
    new_id = math.inf
    all_bad, held_before = _unmatched_sells([*existing, (transaction_date, new_id, "SELL", quantity)])
    bad = all_bad - legacy_bad
    if new_id in bad:
        raise ValueError(
            f"Cannot sell {quantity:g} shares of {symbol} on {transaction_date}: "
            f"only {held_before[new_id]:g} held as of that date"
        )
    if bad:
        victim = next(t for t in existing if t[1] in bad)
        raise ValueError(
            f"Cannot sell {quantity:g} shares of {symbol} on {transaction_date}: it would leave the "
            f"existing SELL of {victim[3]:g} on {victim[0]} without enough shares to match"
        )


def _record(
    symbol: str,
    transaction_type: str,
    quantity: float,
    price: float,
    transaction_date: str,
    fees: float,
    notes: str | None,
    db_path: Path | None,
) -> int:
    quantity = _validate_number("quantity", quantity, allow_zero=False)
    price = _validate_number("price", price, allow_zero=False)
    fees = _validate_number("fees", fees, allow_zero=True)
    symbol = normalize_symbol(symbol)
    transaction_date = _validate_date(transaction_date)

    with get_connection(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO transactions (symbol, transaction_type, quantity, price, fees, transaction_date, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (symbol, transaction_type, quantity, price, fees, transaction_date, notes),
        )
        transaction_id = cursor.lastrowid

    logger.info("Recorded %s #%s for %s", transaction_type, transaction_id, symbol)
    logger.debug("Transaction #%s: %s @ %s on %s", transaction_id, quantity, price, transaction_date)
    assert transaction_id is not None
    return transaction_id


def list_transactions(symbol: str | None = None, db_path: Path | None = None) -> list[dict[str, Any]]:
    query = "SELECT * FROM transactions"
    params: tuple[Any, ...] = ()
    if symbol:
        query += " WHERE symbol = ?"
        params = (normalize_symbol(symbol),)
    query += " ORDER BY transaction_date, id"

    with get_connection(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(row) for row in rows]
