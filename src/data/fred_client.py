# src/data/fred_client.py

import csv
import io
import time
from typing import Any

import requests

from src.config import (
    FRED_API_KEY,
    FRED_BASE_URL,
    FRED_CSV_URL,
    FRED_MAX_RETRIES,
    FRED_RETRY_BACKOFF_SECONDS,
    FRED_RETRYABLE_STATUS_CODES,
)
from src.utils.cache import cache_get, cache_set, is_miss, ttl_for
from src.utils.logging import get_logger, redact_secrets

logger = get_logger("fred_client")


class FREDError(Exception):
    """Raised when the FRED API returns an error or unexpected response."""


class FREDClient:
    """Thin wrapper around the FRED (Federal Reserve Economic Data) API.

    Free, no-cost API — used here to supply macro/economic indicators that
    FMP's Starter tier doesn't provide.

    Two modes: with `FRED_API_KEY` it uses the official JSON API; without one
    it falls back to FRED's public CSV download (same series IDs, full
    history, no key) so macro analysis still works out of the box.
    """

    def __init__(self, api_key: str | None = None, timeout: int = 15):
        self.api_key = api_key or FRED_API_KEY
        self.keyless = not self.api_key
        if self.keyless:
            logger.info("FRED_API_KEY not set — using FRED's keyless public CSV endpoint")
        self.timeout = timeout
        self.session = requests.Session()

    def _get_csv(self, series_id: str) -> list[dict[str, Any]]:
        """Full history for a series via the keyless CSV download, oldest first."""
        last_exc: Exception | None = None
        for attempt in range(1, FRED_MAX_RETRIES + 1):
            try:
                response = self.session.get(FRED_CSV_URL, params={"id": series_id}, timeout=self.timeout)
            except requests.RequestException as exc:
                last_exc = exc
                logger.warning(
                    "FRED CSV %s failed (attempt %d/%d): %s",
                    series_id,
                    attempt,
                    FRED_MAX_RETRIES,
                    redact_secrets(str(exc)),
                )
            else:
                if response.status_code == 200 and response.text.lstrip().lower().startswith(
                    ("observation_date", "date")
                ):
                    rows = csv.reader(io.StringIO(response.text))
                    next(rows, None)
                    out = []
                    for row in rows:
                        if len(row) >= 2 and row[1] not in ("", "."):
                            try:
                                out.append({"date": row[0], "value": float(row[1])})
                            except ValueError:
                                continue
                    return out
                if response.status_code not in FRED_RETRYABLE_STATUS_CODES:
                    raise FREDError(f"FRED series {series_id} not available (status {response.status_code})")
                last_exc = FREDError(f"FRED CSV {series_id} returned status {response.status_code}")
            if attempt < FRED_MAX_RETRIES:
                time.sleep(FRED_RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1)))
        raise FREDError(
            f"FRED CSV {series_id} failed after {FRED_MAX_RETRIES} attempts ({type(last_exc).__name__})"
        ) from None

    def get_history(self, series_id: str, start: str | None = None) -> list[dict[str, Any]]:
        """Full observation history (oldest first), optionally from `start`
        (YYYY-MM-DD). Cached per series; works with or without an API key.
        """
        if not series_id or not series_id.strip():
            raise ValueError("series_id must be a non-empty string")
        series_id = series_id.strip().upper()
        key = {"series": series_id, "mode": "csv" if self.keyless else "api"}
        cached = cache_get("fred", key, ttl_for("macro"))
        if is_miss(cached):
            if self.keyless:
                cached = self._get_csv(series_id)
            else:
                data = self._get("series/observations", {"series_id": series_id, "sort_order": "asc"})
                cached = [
                    {"date": o["date"], "value": float(o["value"])}
                    for o in data.get("observations", [])
                    if o.get("value") not in (None, ".")
                ]
            cache_set("fred", key, cached)
        return [r for r in cached if not start or r["date"] >= start]

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{FRED_BASE_URL}/{path.lstrip('/')}"
        query = {"api_key": self.api_key, "file_type": "json", **(params or {})}

        last_exc: Exception | None = None
        for attempt in range(1, FRED_MAX_RETRIES + 1):
            try:
                response = self.session.get(url, params=query, timeout=self.timeout)
            except requests.RequestException as exc:
                last_exc = exc
                logger.warning(
                    "FRED request to %s failed (attempt %d/%d): %s",
                    path,
                    attempt,
                    FRED_MAX_RETRIES,
                    redact_secrets(str(exc)),
                )
            else:
                if response.status_code == 200:
                    return response.json()

                if response.status_code not in FRED_RETRYABLE_STATUS_CODES:
                    logger.error(
                        "FRED request %s returned status %s: %s",
                        path,
                        response.status_code,
                        response.text[:300],
                    )
                    raise FREDError(f"FRED request {path} returned status {response.status_code}")

                last_exc = FREDError(f"FRED request {path} returned status {response.status_code}")
                logger.warning(
                    "Retryable FRED status %s for %s (attempt %d/%d)",
                    response.status_code,
                    path,
                    attempt,
                    FRED_MAX_RETRIES,
                )

            if attempt < FRED_MAX_RETRIES:
                time.sleep(FRED_RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1)))

        logger.error("FRED request %s failed after %d attempts", path, FRED_MAX_RETRIES)
        raise FREDError(
            f"Request to {path} failed after {FRED_MAX_RETRIES} attempts ({type(last_exc).__name__})"
        ) from None

    def get_series(self, series_id: str, limit: int = 12) -> list[dict[str, Any]]:
        """Most recent `limit` observations for a FRED series, newest first.
        Missing values (FRED uses ".") are dropped.
        """
        if limit < 1:
            raise ValueError(f"limit must be >= 1, got {limit}")

        if self.keyless:
            return list(reversed(self.get_history(series_id)))[:limit]

        data = self._get(
            "series/observations",
            {"series_id": series_id, "sort_order": "desc", "limit": limit},
        )
        observations = data.get("observations", [])
        return [
            {"date": obs["date"], "value": float(obs["value"])}
            for obs in observations
            if obs.get("value") not in (None, ".")
        ]

    def get_latest(self, series_id: str) -> dict[str, Any] | None:
        """Most recent non-missing observation for a series, or None."""
        series = self.get_series(series_id, limit=1)
        return series[0] if series else None
