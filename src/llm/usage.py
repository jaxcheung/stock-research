# src/llm/usage.py
#
# Usage / cost ledger for every LLM call the agents make (SQLite, data/llm_usage.db —
# gitignored). Feeds the budget review: tokens, cache writes/reads, cache-hit rate, list-
# price cost (on a Claude subscription this is the API-equivalent cost, i.e. a measure of
# quota consumed, not a bill), latency and errors — per call, run, agent and model.

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import LLM_USAGE_DB_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    run_id TEXT,
    agent TEXT NOT NULL,
    backend TEXT NOT NULL,
    model TEXT,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    cache_write_tokens INTEGER DEFAULT 0,
    cache_read_tokens INTEGER DEFAULT 0,
    cost_usd REAL DEFAULT 0,
    cost_basis TEXT,
    duration_ms INTEGER,
    is_error INTEGER DEFAULT 0,
    error TEXT,
    prompt_sha256 TEXT
);
"""


@dataclass
class CallUsage:
    agent: str
    backend: str
    model: str | None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    cost_usd: float = 0.0
    cost_basis: str = "list"  # "list" (subscription: API-equivalent) | "billed" (API)
    duration_ms: int | None = None
    is_error: bool = False
    error: str | None = None
    prompt_sha256: str | None = None
    run_id: str | None = None

    @property
    def total_input(self) -> int:
        return self.input_tokens + self.cache_write_tokens + self.cache_read_tokens

    @property
    def cache_hit_rate(self) -> float | None:
        """Share of input tokens served from cache (reads ÷ all input)."""
        return self.cache_read_tokens / self.total_input if self.total_input else None


@contextmanager
def _conn(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    path = db_path or LLM_USAGE_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    try:
        con.execute(_SCHEMA)
        yield con
        con.commit()
    finally:
        con.close()


def record(u: CallUsage, db_path: Path | None = None) -> None:
    with _conn(db_path) as con:
        con.execute(
            "INSERT INTO llm_calls (ts, run_id, agent, backend, model, input_tokens, output_tokens, "
            "cache_write_tokens, cache_read_tokens, cost_usd, cost_basis, duration_ms, is_error, error, "
            "prompt_sha256) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                u.run_id,
                u.agent,
                u.backend,
                u.model,
                u.input_tokens,
                u.output_tokens,
                u.cache_write_tokens,
                u.cache_read_tokens,
                u.cost_usd,
                u.cost_basis,
                u.duration_ms,
                int(u.is_error),
                u.error,
                u.prompt_sha256,
            ),
        )


def calls(db_path: Path | None = None) -> pd.DataFrame:
    with _conn(db_path) as con:
        return pd.read_sql_query("SELECT * FROM llm_calls ORDER BY ts", con)


def summary(group_by: str = "agent", db_path: Path | None = None) -> pd.DataFrame:
    """Budget-review table: calls, tokens, cache-hit rate, cost, latency per `group_by`
    (agent | model | backend | run_id | day)."""
    df = calls(db_path)
    if df.empty:
        return df
    if group_by == "day":
        df["day"] = df["ts"].str[:10]
    g = df.groupby(group_by)
    out = pd.DataFrame(
        {
            "calls": g.size(),
            "errors": g["is_error"].sum(),
            "input": g["input_tokens"].sum(),
            "cacheWrite": g["cache_write_tokens"].sum(),
            "cacheRead": g["cache_read_tokens"].sum(),
            "output": g["output_tokens"].sum(),
            "costUSD": g["cost_usd"].sum(),
            "medianLatencyMs": g["duration_ms"].median(),
        }
    )
    total_in = out["input"] + out["cacheWrite"] + out["cacheRead"]
    out["cacheHitRate"] = (out["cacheRead"] / total_in).where(total_in > 0)
    out["costPerCallUSD"] = out["costUSD"] / out["calls"]
    return out.reset_index()


def usage_from_claude_code(result: dict[str, Any]) -> dict[str, Any]:
    """Map a `claude -p --output-format json` result to ledger fields."""
    u = result.get("usage") or {}
    models = list((result.get("modelUsage") or {}).keys())
    return {
        "model": models[0] if models else None,
        "input_tokens": int(u.get("input_tokens") or 0),
        "output_tokens": int(u.get("output_tokens") or 0),
        "cache_write_tokens": int(u.get("cache_creation_input_tokens") or 0),
        "cache_read_tokens": int(u.get("cache_read_input_tokens") or 0),
        "cost_usd": float(result.get("total_cost_usd") or 0.0),
        "duration_ms": int(result.get("duration_ms") or 0),
    }
