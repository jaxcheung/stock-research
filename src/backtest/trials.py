# src/backtest/trials.py
#
# Persistent ledger of every strategy configuration ever evaluated. The deflated Sharpe ratio
# corrects for multiple testing using the number of trials and the dispersion of their results;
# if that count only covered the current run, running strategies one at a time would reset it.
# Each (strategy, parameters) pair is one trial; re-running the same configuration updates its
# row instead of adding a new trial.

import hashlib
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import BACKTEST_TRIAL_LEDGER_PATH
from src.utils.logging import get_logger

logger = get_logger("backtest.trials")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trials (
    key TEXT PRIMARY KEY,
    strategy TEXT NOT NULL,
    params TEXT NOT NULL,
    info_ratio REAL,
    n_obs INTEGER,
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL
)
"""


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(_SCHEMA)
    return conn


def trial_key(strategy: str, params: dict[str, Any]) -> str:
    blob = json.dumps({"strategy": strategy, **params}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def record(
    strategy: str,
    params: dict[str, Any],
    info_ratio: float,
    n_obs: int,
    path: Path = BACKTEST_TRIAL_LEDGER_PATH,
) -> None:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    key = trial_key(strategy, params)
    try:
        with closing(_connect(path)) as conn, conn:
            conn.execute(
                "INSERT INTO trials VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(key) DO UPDATE SET "
                "info_ratio = excluded.info_ratio, n_obs = excluded.n_obs, last_seen = excluded.last_seen",
                (key, strategy, json.dumps(params, sort_keys=True, default=str), info_ratio, n_obs, now, now),
            )
    except sqlite3.Error as exc:  # the ledger must never break an evaluation
        logger.warning("Trial ledger write failed (%s)", exc)


def load(path: Path = BACKTEST_TRIAL_LEDGER_PATH) -> pd.DataFrame:
    if not path.exists():  # reading never creates the ledger
        return pd.DataFrame(
            columns=["key", "strategy", "params", "info_ratio", "n_obs", "first_seen", "last_seen"]
        )
    try:
        with closing(_connect(path)) as conn:
            return pd.read_sql_query("SELECT * FROM trials ORDER BY first_seen", conn)
    except sqlite3.Error as exc:
        logger.warning("Trial ledger read failed (%s)", exc)
        return pd.DataFrame(
            columns=["key", "strategy", "params", "info_ratio", "n_obs", "first_seen", "last_seen"]
        )
