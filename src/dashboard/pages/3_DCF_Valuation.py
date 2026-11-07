import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd
import streamlit as st

from src.analysis.dcf import live_risk_free_rate, run_dcf
from src.config import (
    DCF_MARKET_RISK_PREMIUM,
    DCF_PROJECTION_YEARS,
    DCF_TERMINAL_GROWTH_RATE,
)
from src.dashboard.ui import chart, current_theme
from src.data.fmp_client import FMPError
from src.reporting.charts import category_bars

st.set_page_config(page_title="DCF Valuation", page_icon="📈", layout="wide")
st.title("DCF Valuation")

if "symbol" not in st.session_state:
    st.session_state["symbol"] = "AAPL"

raw_symbol = st.text_input("Ticker", key="symbol")
symbol = raw_symbol.strip().upper()


@st.cache_data(ttl=3600, show_spinner=False)
def _default_risk_free() -> tuple[float, str]:
    return live_risk_free_rate()


rf_default, rf_source = _default_risk_free()

with st.expander("Assumptions", expanded=True):
    c1, c2, c3 = st.columns(3)
    projection_years = c1.slider("Projection years", 1, 10, DCF_PROJECTION_YEARS)
    risk_free_pct = c2.slider(
        "Risk-free rate (%)",
        0.0,
        25.0,
        round(rf_default * 100, 2),
        0.01,
        help=f"Defaults to {rf_source} — the same rate the Research Report uses.",
    )
    mrp_pct = c3.slider("Market risk premium (%)", 0.0, 20.0, DCF_MARKET_RISK_PREMIUM * 100, 0.25)

    c4, c5 = st.columns(2)
    terminal_growth_pct = c4.slider("Terminal growth (%)", -5.0, 10.0, DCF_TERMINAL_GROWTH_RATE * 100, 0.25)

    use_custom_growth = c5.checkbox("Override FCF growth rate (default: historical CAGR)")
    fcf_growth_rate = None
    if use_custom_growth:
        fcf_growth_pct = c5.slider("FCF growth rate (%)", -50.0, 100.0, 5.0, 1.0)
        fcf_growth_rate = fcf_growth_pct / 100

if st.button("Run DCF", type="primary"):
    try:
        with st.spinner(f"Running DCF for {symbol}..."):
            st.session_state["dcf_result"] = run_dcf(
                symbol,
                projection_years=projection_years,
                # Untouched slider -> pass None so the result records the live FRED source.
                risk_free_rate=None if risk_free_pct == round(rf_default * 100, 2) else risk_free_pct / 100,
                market_risk_premium=mrp_pct / 100,
                terminal_growth_rate=terminal_growth_pct / 100,
                fcf_growth_rate=fcf_growth_rate,
            )
    except (FMPError, ValueError) as exc:
        st.error(str(exc))

result: dict[str, Any] | None = st.session_state.get("dcf_result")
if result and result["symbol"] == symbol:
    upside = result["upsidePct"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Intrinsic Value/Share", f"${result['intrinsicValuePerShare']:.2f}")
    col2.metric("Current Price", f"${result['currentPrice']:.2f}")
    col3.metric(
        "Implied Upside",
        f"{upside * 100:.1f}%" if upside is not None else "n/a",
        delta=f"{upside * 100:.1f}%" if upside is not None else None,
    )

    a = result["assumptions"]
    if a.get("costOfDebtFloored"):
        st.caption(
            f"Cost of debt {a['costOfDebt']:.2%} = risk-free + spread: the derived "
            f"{a['derivedCostOfDebt']:.2%} (interest ÷ debt) reflects legacy coupons, not the marginal cost of borrowing."
        )
    for note in result.get("dataNotes", []):
        st.warning(f"Data quality: {note}")
    tv_share = result.get("terminalValueShare")
    if tv_share is not None:
        msg = f"Terminal value is {tv_share:.0%} of enterprise value."
        if result.get("terminalValueShareFlag"):
            st.warning(msg + " The result is mostly the terminal-growth assumption — treat it as a range.")
        else:
            st.caption(msg)
    if result.get("sanityBandBreach"):
        st.warning(
            "Intrinsic value is outside 0.2×–5× the current price. The Research Report shows this DCF "
            "but does not use it in the price target."
        )

    t = current_theme()
    chart(
        category_bars(
            pd.Series(
                [result["intrinsicValuePerShare"], result["currentPrice"]],
                index=["Intrinsic value", "Current price"],
            ),
            t,
            "Intrinsic value vs current price",
            f"Per share · risk-free rate {result['assumptions']['riskFreeRateSource']}",
        )
    )

    fcf_series = result["projectedFreeCashFlows"]
    chart(
        category_bars(
            pd.Series(fcf_series, index=[f"Year {i + 1}" for i in range(len(fcf_series))]),
            t,
            "Projected free cash flows",
            f"Grown at {result['assumptions']['fcfGrowthRate']:.1%} a year"
            + (" (historical CAGR, clamped)" if result["assumptions"].get("fcfGrowthRateClamped") else ""),
        )
    )

    with st.expander("Full assumptions & intermediate values"):
        st.json(result)
else:
    st.info("Set assumptions and click Run DCF.")
