import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd
import streamlit as st

from src.analysis.indicators import INDICATORS, format_value, load_company
from src.analysis.research_report import generate_research_report, render_research_report
from src.data.fmp_client import FMPClient, FMPError

st.set_page_config(page_title="Research Report", page_icon="📈", layout="wide")
st.title("Equity Research Report")
st.caption(
    "Combines fundamentals, peer comps, DCF valuation, macro backdrop, and the investment "
    "scorecard into one report. Note: fundamental quality and valuation are reported as "
    "separate verdicts — a high-quality business can still be richly valued, and vice versa."
)

if "symbol" not in st.session_state:
    st.session_state["symbol"] = "AAPL"

raw_symbol = st.text_input("Ticker", key="symbol")
symbol = raw_symbol.strip().upper()
max_peers = st.slider("Max peers for comparison", min_value=1, max_value=10, value=5)

if st.button("Generate Report", type="primary"):
    try:
        with st.spinner(f"Generating research report for {symbol}..."):
            report = generate_research_report(symbol, max_peers=max_peers)
            st.session_state["research_report"] = render_research_report(symbol, report=report)
            st.session_state["research_template"] = report["stock"].get("template")
            st.session_state["research_report_symbol"] = symbol
    except (FMPError, ValueError) as exc:
        st.error(str(exc))


@st.cache_data(ttl=3600, show_spinner=False)
def _indicator_history(sym: str) -> pd.DataFrame:
    """Fiscal-year indicator values (rows = years) for the extra-indicator table."""
    return load_company(sym, FMPClient()).values


report_text = st.session_state.get("research_report")
if not (report_text and st.session_state.get("research_report_symbol") == symbol):
    st.info("Enter a ticker and click Generate Report.")
    st.stop()

tpl = st.session_state.get("research_template")
if tpl:
    with st.expander(f"Scoring template — {tpl['template']}", expanded=False):
        st.markdown(tpl["why"])
        st.dataframe(
            pd.DataFrame(
                [{"dimension": d, "scored metrics": ", ".join(ms)} for d, ms in tpl["scored"].items()]
                + [
                    {"dimension": d, "scored metrics": f"not scored — {r}"}
                    for d, r in tpl["unscored"].items()
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        if tpl["unavailable"]:
            st.markdown("**Not available on FMP Starter, so not scored:**")
            st.markdown("\n".join(f"- {u}" for u in tpl["unavailable"]))

st.markdown(report_text)

st.markdown("---")
st.markdown("#### Extra indicators")
st.caption(
    "Display only: indicators you add here are shown for context and do NOT change the template's rating. "
    "The defaults are the indicators this industry template recommends watching."
)
options = sorted(k for k, i in INDICATORS.items() if not i.market)
defaults = [k for k in (tpl or {}).get("display", []) if k in options]
picked = st.multiselect(
    "Indicators",
    options,
    default=defaults,
    format_func=lambda k: f"{INDICATORS[k].label} ({INDICATORS[k].category})",
    key=f"extra_{symbol}",
)
if picked:
    try:
        hist = _indicator_history(symbol)
    except (FMPError, ValueError) as exc:
        st.error(str(exc))
    else:
        cols = [k for k in picked if k in hist.columns]
        years = list(hist.index[-5:])
        shown = pd.DataFrame(
            {str(y): [format_value(hist.at[y, k], INDICATORS[k].fmt) for k in cols] for y in years},
            index=[INDICATORS[k].label for k in cols],
        )
        st.dataframe(shown, width="stretch")
        st.caption("; ".join(f"**{INDICATORS[k].label}**: {INDICATORS[k].description}" for k in cols))
