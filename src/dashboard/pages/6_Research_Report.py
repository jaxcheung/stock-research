import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import streamlit as st

from src.analysis.research_report import render_research_report
from src.data.fmp_client import FMPError

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
            st.session_state["research_report"] = render_research_report(symbol, max_peers=max_peers)
            st.session_state["research_report_symbol"] = symbol
    except (FMPError, ValueError) as exc:
        st.error(str(exc))

report_text = st.session_state.get("research_report")
if report_text and st.session_state.get("research_report_symbol") == symbol:
    st.markdown(report_text)
else:
    st.info("Enter a ticker and click Generate Report.")
