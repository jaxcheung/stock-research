# src/utils/cache.py
#
# Small on-disk TTL cache for API responses (FMP, FRED, World Bank). Keeps
# batch analyses (peer groups, industry screens, multi-series macro pulls)
# inside Starter-tier rate limits and makes re-runs near-instant.
#
# Files live under data/cache/ (gitignored). Keys are SHA-256 hashes of the
# request, so API keys never appear in filenames; callers must still exclude
# secrets from the key material they pass in.

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from src.config import CACHE_DIR, CACHE_ENABLED, CACHE_TTL_SECONDS
from src.utils.logging import get_logger

logger = get_logger("cache")

_MISS = object()


def _path_for(namespace: str, key_material: dict[str, Any]) -> Path:
    digest = hashlib.sha256(json.dumps(key_material, sort_keys=True, default=str).encode()).hexdigest()
    return CACHE_DIR / namespace / f"{digest[:32]}.json"


def ttl_for(category: str) -> int:
    return CACHE_TTL_SECONDS.get(category, CACHE_TTL_SECONDS["default"])


def cache_get(namespace: str, key_material: dict[str, Any], ttl_seconds: int) -> Any:
    """Cached value, or the module-level `_MISS` sentinel if absent/expired/disabled."""
    if not CACHE_ENABLED or ttl_seconds <= 0:
        return _MISS
    path = _path_for(namespace, key_material)
    try:
        if time.time() - path.stat().st_mtime > ttl_seconds:
            return _MISS
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return _MISS


def cache_set(namespace: str, key_material: dict[str, Any], value: Any) -> None:
    if not CACHE_ENABLED:
        return
    path = _path_for(namespace, key_material)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(value, default=str))
        tmp.replace(path)  # atomic on POSIX — no half-written cache files
    except (OSError, TypeError) as exc:
        logger.warning("Cache write failed for %s: %s", namespace, exc)


def is_miss(value: Any) -> bool:
    return value is _MISS


def clear_cache(namespace: str | None = None) -> int:
    """Delete cached files (one namespace, or all). Returns files removed."""
    root = CACHE_DIR / namespace if namespace else CACHE_DIR
    removed = 0
    for f in root.rglob("*.json") if root.exists() else []:
        f.unlink(missing_ok=True)
        removed += 1
    return removed
