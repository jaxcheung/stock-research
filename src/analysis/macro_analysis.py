# src/analysis/macro_analysis.py
#
# Macro/economic backdrop sourced from FRED, since FMP's own economic
# indicators endpoints are Premium-gated on Starter tier. Used to give a
# research report real economic context (growth, inflation, rates, yield
# curve) rather than analyzing a stock in a vacuum.

from datetime import date
from typing import Any

from src.config import FRED_SERIES
from src.data.fred_client import FREDClient
from src.utils.logging import get_logger

logger = get_logger("macro_analysis")


def _cpi_yoy(client: FREDClient) -> dict[str, Any] | None:
    """YoY CPI inflation: latest observation vs the one dated exactly 12 months earlier.

    Matched by date, not position — FRED drops missing months (e.g. Oct-2025),
    so "13th most recent row" can silently be a 13-month change.
    """
    series = client.get_series(FRED_SERIES["cpi_index"], limit=24)
    if not series:
        logger.warning("No CPI history available to compute YoY inflation")
        return None
    latest = series[0]
    latest_date = date.fromisoformat(latest["date"])
    target = latest_date.replace(year=latest_date.year - 1)
    year_ago = next((o for o in series if date.fromisoformat(o["date"]) == target), None)
    if year_ago is None or not year_ago["value"]:
        logger.warning("CPI observation for %s missing; YoY inflation unavailable", target.isoformat())
        return None
    return {
        "date": latest["date"],
        "value": latest["value"] / year_ago["value"] - 1,
        "baseDate": year_ago["date"],
    }


def get_macro_backdrop(client: FREDClient | None = None) -> dict[str, Any]:
    client = client or FREDClient()

    real_gdp_growth = client.get_latest(FRED_SERIES["real_gdp_growth"])
    cpi_yoy = _cpi_yoy(client)
    unemployment_rate = client.get_latest(FRED_SERIES["unemployment_rate"])
    fed_funds_rate = client.get_latest(FRED_SERIES["fed_funds_rate"])
    treasury_10y = client.get_latest(FRED_SERIES["treasury_10y"])
    yield_curve = client.get_latest(FRED_SERIES["yield_curve_10y2y"])

    yield_curve_inverted = yield_curve is not None and yield_curve["value"] < 0

    return {
        "realGdpGrowth": real_gdp_growth,
        "cpiYoY": cpi_yoy,
        "unemploymentRate": unemployment_rate,
        "fedFundsRate": fed_funds_rate,
        "treasury10Y": treasury_10y,
        "yieldCurve10Y2Y": yield_curve,
        "yieldCurveInverted": yield_curve_inverted,
    }


def summarize_macro(client: FREDClient | None = None) -> str:
    """Render get_macro_backdrop() output as a short markdown section."""
    m = get_macro_backdrop(client=client)

    def pct_points(entry: dict[str, Any] | None) -> str:
        """FRED series already expressed in percentage points (e.g. 5.33 = 5.33%)."""
        return f"{entry['value']:.2f}%" if entry else "n/a"

    def pct_ratio(entry: dict[str, Any] | None) -> str:
        """Decimal fraction (e.g. 0.031 = 3.1%) — our own computed ratios."""
        return f"{entry['value']:.2%}" if entry else "n/a"

    yield_curve_note = (
        "⚠️ Inverted — historically a leading recession indicator"
        if m["yieldCurveInverted"]
        else "Normal (not inverted)"
    )

    return "\n".join(
        [
            "## Macro Backdrop",
            f"- Real GDP growth (SAAR): {pct_points(m['realGdpGrowth'])} (as of {m['realGdpGrowth']['date'] if m['realGdpGrowth'] else 'n/a'})",
            f"- CPI inflation (YoY): {pct_ratio(m['cpiYoY'])}",
            f"- Unemployment rate: {pct_points(m['unemploymentRate'])}",
            f"- Fed funds rate: {pct_points(m['fedFundsRate'])}",
            f"- 10Y Treasury yield: {pct_points(m['treasury10Y'])}",
            f"- Yield curve (10Y-2Y): {pct_points(m['yieldCurve10Y2Y'])} — {yield_curve_note}",
        ]
    )
