# src/data/fmp_client.py

import time
from typing import Any

import requests

from src.config import (
    FMP_API_KEY,
    FMP_BASE_URL,
    FMP_MAX_CALLS_PER_MINUTE,
    FMP_MAX_RETRIES,
    FMP_RETRY_BACKOFF_SECONDS,
    FMP_RETRYABLE_STATUS_CODES,
)
from src.utils.cache import cache_get, cache_set, is_miss, ttl_for
from src.utils.logging import get_logger, redact_secrets

logger = get_logger("fmp_client")

# Cache category per endpoint (TTLs in config.CACHE_TTL_SECONDS). Anything
# not listed uses "default".
_CACHE_CATEGORY = {
    "quote": "quote",
    "income-statement": "statements",
    "balance-sheet-statement": "statements",
    "cash-flow-statement": "statements",
    "ratios": "statements",
    "key-metrics": "statements",
    "ratios-ttm": "statements",
    "key-metrics-ttm": "statements",
    "profile": "statements",
    "stock-peers": "statements",
    "sector-pe-snapshot": "market",
    "industry-pe-snapshot": "market",
    "historical-sector-pe": "market",
    "historical-industry-pe": "market",
    "company-screener": "market",
    "historical-price-eod/light": "market",
    "available-industries": "statements",
    "available-sectors": "statements",
    "available-countries": "statements",
}


class FMPError(Exception):
    """Raised when the FMP API returns an error or unexpected response."""


def normalize_symbol(symbol: str) -> str:
    if not symbol or not symbol.strip():
        raise ValueError("symbol must be a non-empty string")
    return symbol.strip().upper()


def _validate_limit(limit: int) -> None:
    if limit < 1:
        raise ValueError(f"limit must be >= 1, got {limit}")


class FMPClient:
    """Thin wrapper around the Financial Modeling Prep `/stable` API (Starter tier).

    Verified available on Starter (2026-11-05): statements (10y annual, 40
    quarters), ratios/key-metrics incl. TTM, daily price history, sector &
    industry P/E snapshots and history, company screener, treasury rates.
    Verified 402 (Premium): batch-quote, sp500-constituent, foreign-exchange
    tickers. Successful responses are cached on disk (see src/utils/cache.py);
    errors are never cached.
    """

    def __init__(self, api_key: str | None = None, timeout: int = 10):
        self.api_key = api_key or FMP_API_KEY
        if not self.api_key:
            raise FMPError("FMP_API_KEY is not set (expected in environment or .env)")
        self.timeout = timeout
        self.session = requests.Session()
        self._last_call = 0.0

    def _throttle(self) -> None:
        """Space live calls to stay under FMP_MAX_CALLS_PER_MINUTE (bulk syncs)."""
        min_interval = 60.0 / FMP_MAX_CALLS_PER_MINUTE
        wait = self._last_call + min_interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        path = path.lstrip("/")
        # Cache key deliberately excludes the API key.
        cache_key = {"path": path, "params": params or {}}
        ttl = ttl_for(_CACHE_CATEGORY.get(path, "default"))
        cached = cache_get("fmp", cache_key, ttl)
        if not is_miss(cached):
            return cached
        data = self._request(path, params)
        cache_set("fmp", cache_key, data)
        return data

    def _request(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{FMP_BASE_URL}/{path}"
        query = {"apikey": self.api_key, **(params or {})}

        last_exc: Exception | None = None
        for attempt in range(1, FMP_MAX_RETRIES + 1):
            self._throttle()
            try:
                response = self.session.get(url, params=query, timeout=self.timeout)
            except requests.RequestException as exc:
                last_exc = exc
                logger.warning(
                    "Request to %s failed (attempt %d/%d): %s",
                    path,
                    attempt,
                    FMP_MAX_RETRIES,
                    redact_secrets(str(exc)),
                )
            else:
                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, dict) and "Error Message" in data:
                        logger.error("FMP error for %s: %s", path, data["Error Message"])
                        raise FMPError(data["Error Message"])
                    return data

                if response.status_code not in FMP_RETRYABLE_STATUS_CODES:
                    logger.error(
                        "FMP request %s returned status %s: %s",
                        path,
                        response.status_code,
                        response.text[:300],
                    )
                    raise FMPError(f"FMP request {path} returned status {response.status_code}")

                last_exc = FMPError(f"FMP request {path} returned status {response.status_code}")
                logger.warning(
                    "Retryable FMP status %s for %s (attempt %d/%d)",
                    response.status_code,
                    path,
                    attempt,
                    FMP_MAX_RETRIES,
                )

            if attempt < FMP_MAX_RETRIES:
                time.sleep(FMP_RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1)))

        logger.error("FMP request %s failed after %d attempts", path, FMP_MAX_RETRIES)
        raise FMPError(
            f"Request to {path} failed after {FMP_MAX_RETRIES} attempts ({type(last_exc).__name__})"
        ) from None

    def get_quote(self, symbol: str) -> dict[str, Any]:
        data = self._get("quote", {"symbol": normalize_symbol(symbol)})
        return data[0] if data else {}

    def get_profile(self, symbol: str) -> dict[str, Any]:
        data = self._get("profile", {"symbol": normalize_symbol(symbol)})
        return data[0] if data else {}

    def get_income_statement(
        self, symbol: str, period: str = "annual", limit: int = 5
    ) -> list[dict[str, Any]]:
        _validate_limit(limit)
        return self._get(
            "income-statement",
            {"symbol": normalize_symbol(symbol), "period": period, "limit": limit},
        )

    def get_balance_sheet(self, symbol: str, period: str = "annual", limit: int = 5) -> list[dict[str, Any]]:
        _validate_limit(limit)
        return self._get(
            "balance-sheet-statement",
            {"symbol": normalize_symbol(symbol), "period": period, "limit": limit},
        )

    def get_cash_flow(self, symbol: str, period: str = "annual", limit: int = 5) -> list[dict[str, Any]]:
        _validate_limit(limit)
        return self._get(
            "cash-flow-statement",
            {"symbol": normalize_symbol(symbol), "period": period, "limit": limit},
        )

    def get_ratios(self, symbol: str, period: str = "annual", limit: int = 5) -> list[dict[str, Any]]:
        _validate_limit(limit)
        return self._get(
            "ratios",
            {"symbol": normalize_symbol(symbol), "period": period, "limit": limit},
        )

    def get_key_metrics(self, symbol: str, period: str = "annual", limit: int = 5) -> list[dict[str, Any]]:
        _validate_limit(limit)
        return self._get(
            "key-metrics",
            {"symbol": normalize_symbol(symbol), "period": period, "limit": limit},
        )

    def get_stock_peers(self, symbol: str) -> list[dict[str, Any]]:
        """Companies in the same sector/exchange with a similar market cap."""
        return self._get("stock-peers", {"symbol": normalize_symbol(symbol)})

    def get_stock_news(self, tickers: list[str] | None = None, limit: int = 50) -> list[dict[str, Any]]:
        _validate_limit(limit)
        if tickers:
            symbols = ",".join(normalize_symbol(t) for t in tickers)
            return self._get("news/stock", {"symbols": symbols, "limit": limit})
        return self._get("news/stock-latest", {"limit": limit})

    # --- TTM, price history, market-level PE (all verified on Starter tier) ---

    def get_ratios_ttm(self, symbol: str) -> dict[str, Any]:
        data = self._get("ratios-ttm", {"symbol": normalize_symbol(symbol)})
        return data[0] if data else {}

    def get_key_metrics_ttm(self, symbol: str) -> dict[str, Any]:
        data = self._get("key-metrics-ttm", {"symbol": normalize_symbol(symbol)})
        return data[0] if data else {}

    def get_historical_prices(self, symbol: str, start: str | None = None) -> list[dict[str, Any]]:
        """Daily close history (newest first): [{date, price, volume}, ...]."""
        params: dict[str, Any] = {"symbol": normalize_symbol(symbol)}
        if start:
            params["from"] = start
        return self._get("historical-price-eod/light", params)

    def get_sector_pe_snapshot(self, date: str, exchange: str | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"date": date}
        if exchange:
            params["exchange"] = exchange
        return self._get("sector-pe-snapshot", params)

    def get_industry_pe_snapshot(self, date: str, exchange: str | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"date": date}
        if exchange:
            params["exchange"] = exchange
        return self._get("industry-pe-snapshot", params)

    def get_historical_sector_pe(
        self, sector: str, start: str, end: str, exchange: str | None = None
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"sector": sector, "from": start, "to": end}
        if exchange:
            params["exchange"] = exchange
        return self._get("historical-sector-pe", params)

    def get_historical_industry_pe(
        self, industry: str, start: str, end: str, exchange: str | None = None
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"industry": industry, "from": start, "to": end}
        if exchange:
            params["exchange"] = exchange
        return self._get("historical-industry-pe", params)

    def screen_companies(self, limit: int = 50, **filters: Any) -> list[dict[str, Any]]:
        """FMP company screener, e.g. screen_companies(industry="Auto - Manufacturers",
        country="CN", marketCapMoreThan=1e9). Filter names follow FMP's API.
        """
        _validate_limit(limit)
        params = {k: v for k, v in filters.items() if v is not None}
        params.setdefault("isActivelyTrading", "true")
        return self._get("company-screener", {**params, "limit": limit})

    # --- Reference lists (drive UI pickers; cached a day) ---

    def list_industries(self) -> list[str]:
        return sorted(r["industry"] for r in self._get("available-industries", {}) if r.get("industry"))

    def list_sectors(self) -> list[str]:
        return sorted(r["sector"] for r in self._get("available-sectors", {}) if r.get("sector"))

    def list_countries(self) -> list[str]:
        return sorted(r["country"] for r in self._get("available-countries", {}) if r.get("country"))
