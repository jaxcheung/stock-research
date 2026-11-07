# src/data/worldbank_client.py
#
# World Bank Indicators API — keyless, annual, ~200 countries. Fills gaps
# FRED doesn't cover well for non-US economies (e.g. China's bank lending
# rate, real GDP growth). Same retry/backoff + cache pattern as the other clients.

import time
from typing import Any

import requests

from src.config import (
    WORLD_BANK_BASE_URL,
    WORLD_BANK_MAX_RETRIES,
    WORLD_BANK_RETRY_BACKOFF_SECONDS,
    WORLD_BANK_RETRYABLE_STATUS_CODES,
)
from src.utils.cache import cache_get, cache_set, is_miss, ttl_for
from src.utils.logging import get_logger

logger = get_logger("worldbank_client")


class WorldBankError(Exception):
    """Raised when the World Bank API returns an error or unexpected response."""


class WorldBankClient:
    def __init__(self, timeout: int = 15):
        self.timeout = timeout
        self.session = requests.Session()

    def get_indicator(self, country_iso3: str, indicator: str) -> list[dict[str, Any]]:
        """Annual observations (oldest first) as [{date: 'YYYY-12-31', value}, ...]."""
        country = (country_iso3 or "").strip().upper()
        indicator = (indicator or "").strip()
        if len(country) != 3 or not indicator:
            raise ValueError(f"expected ISO3 country and indicator, got {country_iso3!r}, {indicator!r}")

        key = {"country": country, "indicator": indicator}
        cached = cache_get("worldbank", key, ttl_for("macro"))
        if not is_miss(cached):
            return cached

        url = f"{WORLD_BANK_BASE_URL}/country/{country}/indicator/{indicator}"
        params: dict[str, str | int] = {"format": "json", "per_page": 20000}
        last_exc: Exception | None = None
        for attempt in range(1, WORLD_BANK_MAX_RETRIES + 1):
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                last_exc = exc
                logger.warning("World Bank %s/%s failed (attempt %d): %s", country, indicator, attempt, exc)
            else:
                if response.status_code == 200:
                    payload = response.json()
                    if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
                        message = (
                            payload[0].get("message") if isinstance(payload, list) and payload else payload
                        )
                        raise WorldBankError(f"World Bank {country}/{indicator}: no data ({message})")
                    rows = [
                        {"date": f"{r['date']}-12-31", "value": float(r["value"])}
                        for r in payload[1]
                        if r.get("value") is not None and str(r.get("date", "")).isdigit()
                    ]
                    rows.sort(key=lambda r: str(r["date"]))
                    cache_set("worldbank", key, rows)
                    return rows
                if response.status_code not in WORLD_BANK_RETRYABLE_STATUS_CODES:
                    raise WorldBankError(
                        f"World Bank {country}/{indicator} returned status {response.status_code}"
                    )
                last_exc = WorldBankError(f"status {response.status_code}")
            if attempt < WORLD_BANK_MAX_RETRIES:
                time.sleep(WORLD_BANK_RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1)))
        raise WorldBankError(f"World Bank {country}/{indicator} failed after retries") from last_exc
