# src/analysis/data_quality.py
#
# Deterministic checks on statement data before it feeds an analysis. First
# check: stub rows — FMP sometimes publishes a latest-period statement with a
# handful of fields filled and the rest zero (RJF FY2026 balance sheet: 3
# non-zero fields, total assets $155m vs ~$85bn a year earlier, equity 0). Read
# as real, that produced "negative shareholders' equity ($0.00)" and a 10/100
# health score. A stub is never silently "fixed": the caller falls back to the
# prior period and must surface the returned note.

from typing import Any

from src.config import STUB_STATEMENT_ASSET_RATIO


def is_stub_balance(row: dict[str, Any], prior: dict[str, Any] | None) -> str | None:
    """Reason string if `row` looks like a stub, else None."""
    assets = row.get("totalAssets") or 0.0
    equity = row.get("totalStockholdersEquity") or 0.0
    liabilities = row.get("totalLiabilities") or 0.0
    if assets > 0 and equity == 0 and liabilities == 0:
        return "equity and liabilities are both zero"
    prior_assets = (prior or {}).get("totalAssets") or 0.0
    if prior_assets > 0 and assets < STUB_STATEMENT_ASSET_RATIO * prior_assets:
        return f"total assets {assets:,.0f} are under {STUB_STATEMENT_ASSET_RATIO:.0%} of the prior period's {prior_assets:,.0f}"
    return None


def latest_complete_balance(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], str | None]:
    """(balance sheet to use, data-quality note or None). `rows` is newest-first."""
    if not rows:
        return {}, None
    latest, prior = rows[0], rows[1] if len(rows) > 1 else None
    why = is_stub_balance(latest, prior)
    if why and prior:
        return prior, (
            f"Latest balance sheet ({latest.get('date')}) looks like an incomplete stub ({why}); "
            f"using {prior.get('date')} instead."
        )
    if why:
        return (
            latest,
            f"Latest balance sheet ({latest.get('date')}) looks incomplete ({why}); no prior period to fall back to.",
        )
    return latest, None
