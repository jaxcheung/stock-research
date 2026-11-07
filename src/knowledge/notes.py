# src/knowledge/notes.py

from pathlib import Path
from typing import Any

from src.data.fmp_client import normalize_symbol
from src.knowledge.db import get_connection
from src.utils.logging import get_logger

logger = get_logger("knowledge.notes")


def add_quote(
    text: str,
    author: str | None = None,
    source: str | None = None,
    category: str | None = None,
    symbol: str | None = None,
    context: str | None = None,
    db_path: Path | None = None,
) -> int:
    if not text or not text.strip():
        raise ValueError("text must be a non-empty string")
    symbol = normalize_symbol(symbol) if symbol else None

    with get_connection(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO quotes (text, author, source, category, symbol, context)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (text.strip(), author, source, category, symbol, context),
        )
        quote_id = cursor.lastrowid

    logger.info("Added quote #%s (author=%s, category=%s)", quote_id, author, category)
    assert quote_id is not None
    return quote_id


def list_quotes(
    category: str | None = None,
    symbol: str | None = None,
    author: str | None = None,
    search: str | None = None,
    db_path: Path | None = None,
) -> list[dict[str, Any]]:
    clauses = []
    params: list[Any] = []

    if category:
        clauses.append("category = ?")
        params.append(category)
    if symbol:
        clauses.append("symbol = ?")
        params.append(normalize_symbol(symbol))
    if author:
        clauses.append("author = ?")
        params.append(author)
    if search:
        clauses.append(
            "(text LIKE ? ESCAPE '\\' OR context LIKE ? ESCAPE '\\' "
            "OR source LIKE ? ESCAPE '\\' OR author LIKE ? ESCAPE '\\')"
        )
        escaped = search.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        like = f"%{escaped}%"
        params.extend([like, like, like, like])

    query = "SELECT * FROM quotes"
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY created_at DESC"

    with get_connection(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def get_random_quote(db_path: Path | None = None) -> dict[str, Any] | None:
    with get_connection(db_path) as conn:
        row = conn.execute("SELECT * FROM quotes ORDER BY RANDOM() LIMIT 1").fetchone()
    return dict(row) if row else None


def delete_quote(quote_id: int, db_path: Path | None = None) -> bool:
    with get_connection(db_path) as conn:
        cursor = conn.execute("DELETE FROM quotes WHERE id = ?", (quote_id,))
    deleted = cursor.rowcount > 0
    if deleted:
        logger.info("Deleted quote #%s", quote_id)
    return deleted
