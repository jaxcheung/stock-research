import html
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import streamlit as st

from src.analysis.relative_valuation import RelativeValuationReport, run_relative_valuation
from src.dashboard import ui
from src.data.fmp_client import FMPError
from src.reporting import builders as b

t = ui.setup_page("Relative Valuation", "⚖️")
ui.page_header(
    "Quant Lab",
    "Relative valuation — P/E in context",
    "A stock's multiple against its industry peers, size-matched sector peers, FMP industry and sector "
    "aggregates, the US large-cap market and its own history — with growth and profitability context to judge "
    "whether a premium is earned. Output is a **relative-valuation reading**, not an investment rating.",
)


@st.cache_data(ttl=3600, show_spinner=False)
def _run(symbol: str, max_peers: int, years: int, market: bool) -> RelativeValuationReport:
    return run_relative_valuation(symbol, max_peers=max_peers, history_years=years, include_market=market)


if "symbol" not in st.session_state:
    st.session_state["symbol"] = "AAPL"

with st.form("relval"):
    c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
    sym = c1.text_input("Ticker", key="symbol")
    max_peers = c2.slider("Max industry peers", 3, 20, 10)
    years = c3.slider("History (years)", 2, 10, 5)
    market = c4.toggle(
        "US market benchmark",
        value=True,
        help="First run builds a 30-company market proxy (~1 min), then cached.",
    )
    submitted = st.form_submit_button("Run analysis", type="primary")

if submitted:
    try:
        with st.spinner(f"Building peer set and P/E history for {sym.strip().upper()}…"):
            st.session_state["relval_result"] = _run(sym.strip().upper(), max_peers, years, market)
    except (ValueError, FMPError) as exc:
        st.error(str(exc))

rep: RelativeValuationReport | None = st.session_state.get("relval_result")
if rep is None:
    ui.empty_state("Enter a ticker and Run analysis.")
    st.stop()

st.markdown(
    "".join(
        f"<span class='rp-tag'>{html.escape(str(x))}</span>" for x in (rep.sector, rep.industry, rep.exchange)
    )
    + f"<b>{html.escape(str(rep.company))}</b>",
    unsafe_allow_html=True,
)
ui.kpi_row(b.relval_kpis(rep))
ui.findings(rep.findings)

tabs = st.tabs(["Benchmarks", "Peers", "History", "Data"])
with tabs[0]:
    figs = b.relval_figures(rep, t, show_industry=st.session_state.get("rv_ind", False))
    ui.chart(figs["benchmarks"])
    ui.table(b.benchmark_table(rep))
    st.caption("● = primary benchmark used for the verdict.")
with tabs[1]:
    if "scatter_growth" in figs:
        metric = st.segmented_control(
            "Compare P/E against", ["EPS growth", "Return on equity"], default="EPS growth"
        )
        ui.chart(figs["scatter_roe" if metric == "Return on equity" else "scatter_growth"])
        ui.table(b.peer_table(rep))
    else:
        st.info("No peers available.")
with tabs[2]:
    if "history" in figs:
        st.toggle(
            "Show FMP industry P/E",
            key="rv_ind",
            help="FMP industry aggregates can be extreme when a mega-cap or near-zero earner dominates.",
        )
        ui.chart(figs["history"])
        ui.chart(figs["relative"])
    else:
        st.info("P/E history unavailable for this symbol.")
with tabs[3]:
    if not rep.history.empty:
        ui.table(rep.history.round(2))
        st.download_button(
            "Download P/E history (CSV)",
            rep.history.to_csv().encode(),
            f"{rep.symbol}_pe_history.csv",
            "text/csv",
        )
    if not rep.market_constituents.empty:
        st.markdown("##### US large-cap market proxy constituents")
        ui.table(
            rep.market_constituents[["symbol", "company", "sector", "marketCap", "pe"]].set_index("symbol")
        )

ui.report_footer(
    rep.methodology,
    rep.caveats,
    [
        "FMP ratios-ttm, key-metrics-ttm, statements, profile, screener",
        "FMP sector/industry P/E snapshots & history",
        "FMP daily price history",
    ],
)
ui.download_report(
    b.relative_valuation_html(rep), f"{rep.symbol}_relative_valuation_{date.today():%Y%m%d}.html"
)
