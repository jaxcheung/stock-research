# src/reporting/formatting.py
#
# Display formatting for money and indicator values. Money always carries its
# currency: "$" for USD, the local symbol/code otherwise (a JPY figure is never
# shown with a "$").

import math
from typing import Any

from src.analysis.indicators import format_value

CCY_SYMBOL = {
    "USD": "$",
    "EUR": "€",
    "GBP": "£",
    "JPY": "¥",
    "CNY": "CN¥",
    "HKD": "HK$",
    "TWD": "NT$",
    "KRW": "₩",
    "INR": "₹",
    "CAD": "C$",
    "AUD": "A$",
    "SGD": "S$",
    "BRL": "R$",
    "MXN": "MX$",
}


def ccy_symbol(ccy: str | None) -> str:
    if not ccy:
        return "$"
    return CCY_SYMBOL.get(ccy, f"{ccy} ")


def _bad(v: Any) -> bool:
    return v is None or not isinstance(v, int | float) or not math.isfinite(v)


def money(v: Any, ccy: str | None = "USD", decimals: int = 1) -> str:
    """$5.10tn · $394.3bn · $12.4m · $950k · $12.35 (sign before the symbol)."""
    if _bad(v):
        return "—"
    sym, sign, a = ccy_symbol(ccy), "-" if v < 0 else "", abs(float(v))
    for div, suf in ((1e12, "tn"), (1e9, "bn"), (1e6, "m"), (1e3, "k")):
        if a >= div:
            return f"{sign}{sym}{a / div:,.{decimals if div > 1e3 else 0}f}{suf}"
    return f"{sign}{sym}{a:,.2f}"


def price(v: Any, ccy: str | None = "USD") -> str:
    return "—" if _bad(v) else f"{ccy_symbol(ccy)}{float(v):,.2f}"


def value(v: Any, fmt: str) -> str:
    return format_value(None if _bad(v) else float(v), fmt)


def plotly_tick(fmt: str) -> dict[str, str]:
    """Axis tick formatting for an indicator format."""
    return {
        "pct": {"tickformat": ".0%"},
        "x": {"tickformat": ".1f", "ticksuffix": "×"},
        "days": {"tickformat": ".0f", "ticksuffix": " d"},
    }.get(fmt, {"tickformat": ".2f"})


def hover(fmt: str, axis: str = "y") -> str:
    """Plotly hovertemplate fragment, e.g. hover("pct") -> "%{y:.1%}"."""
    spec = {"pct": ":.1%", "x": ":.1f", "days": ":.0f"}.get(fmt, ":.2f")
    return "%{" + axis + spec + "}" + {"x": "×", "days": " d"}.get(fmt, "")
