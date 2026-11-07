import html
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import streamlit as st

from src.analysis.rate_sensitivity import RateSensitivityReport, run_rate_sensitivity
from src.config import MACRO_SERIES_CATALOG
from src.dashboard import ui
from src.data.fmp_client import FMPClient, FMPError
from src.reporting import builders as b

t = ui.setup_page("Rate Sensitivity", "🏦")
ui.page_header(
    "Quant Lab",
    "Industry rate sensitivity",
    "How do policy and market interest rates feed through to an industry's cost of debt? Effective borrowing "
    "rates from company financials, matched to a macro benchmark over each fiscal year, with pass-through "
    "estimated on a company panel and stress-tested under rate shocks.",
)

RATE_SERIES = {
    k: v for k, v in MACRO_SERIES_CATALOG.items() if v["unit"] == "pct" and "rate" in v["label"].lower()
}
COUNTRIES = {
    "CN": "China",
    "US": "United States",
    "JP": "Japan",
    "DE": "Germany",
    "KR": "South Korea",
    "GB": "United Kingdom",
    "": "Any",
}


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def _industries() -> list[str]:
    try:
        return FMPClient().list_industries()
    except FMPError:
        return ["Auto - Manufacturers", "Auto - Parts", "Real Estate - Development", "Steel", "Airlines"]


@st.cache_data(ttl=3600, show_spinner=False)
def _run(
    bench: str, symbols: tuple[str, ...], industry: str | None, country: str | None, years: int, max_cos: int
) -> RateSensitivityReport:
    return run_rate_sensitivity(
        bench,
        symbols=list(symbols) or None,
        industry=industry,
        country=country,
        years=years,
        max_companies=max_cos,
    )


with st.form("rate_sens"):
    mode = st.segmented_control(
        "Universe", ["Screen an industry", "Custom tickers"], default="Screen an industry"
    )
    c1, c2, c3 = st.columns([3, 2, 3])
    inds = _industries()
    industry = c1.selectbox(
        "Industry (FMP)",
        inds,
        index=inds.index("Auto - Manufacturers") if "Auto - Manufacturers" in inds else 0,
    )
    country = c2.selectbox("Company country", list(COUNTRIES), format_func=lambda c: COUNTRIES[c])
    tickers = c3.text_input(
        "Custom tickers (comma-separated)",
        "GM, F, TSLA, STLA, RIVN, LCID",
        help="Used when Universe = Custom tickers. US-listed symbols (incl. ADRs).",
    )
    c4, c5, c6 = st.columns([3, 2, 2])
    bench = c4.selectbox(
        "Benchmark rate",
        list(RATE_SERIES),
        format_func=lambda k: f"{RATE_SERIES[k]['label']} · {RATE_SERIES[k]['region']}",
    )
    years = c5.slider("Fiscal years", 3, 10, 9)
    max_cos = c6.slider("Max companies", 3, 25, 15)
    submitted = st.form_submit_button("Run analysis", type="primary")

if submitted:
    custom = mode == "Custom tickers"
    syms = tuple(s.strip().upper() for s in tickers.split(",") if s.strip()) if custom else ()
    try:
        with st.spinner("Pulling 10 years of statements and benchmark history…"):
            st.session_state["rate_sens_result"] = _run(
                bench,
                syms,
                None if custom else industry,
                None if custom else (country or None),
                years,
                max_cos,
            )
    except (ValueError, FMPError) as exc:
        st.error(str(exc))

rep: RateSensitivityReport | None = st.session_state.get("rate_sens_result")
if rep is None:
    ui.empty_state("Example: Auto - Manufacturers · China · China 3M interbank rate — then Run analysis.")
    st.stop()

st.markdown(
    f"<span class='rp-tag'>Universe</span>{html.escape(rep.universe_description)}", unsafe_allow_html=True
)
ui.kpi_row(b.rate_kpis(rep))
ui.findings(rep.findings)
figs = b.rate_figures(rep, t)

tabs = st.tabs(["Borrowing costs", "Companies", "Pass-through & scenarios", "Data"])
with tabs[0]:
    ui.chart(figs["rates"])
    ui.chart(figs["spread"])
    st.markdown("##### Industry by fiscal year")
    ui.table(b.rate_industry_table(rep))
with tabs[1]:
    ui.chart(figs["coverage"])
    st.markdown("##### Latest fiscal year")
    ui.table(b.rate_company_table(rep))
with tabs[2]:
    pt = b.passthrough_table(rep)
    if pt is not None:
        st.markdown("##### Pass-through regression — Δ effective rate on Δ benchmark")
        ui.table(pt)
        st.caption("Pooled panel, first differences, company-clustered (CR1) standard errors.")
    if "scenarios" in figs:
        ui.chart(figs["scenarios"])
        ui.table(
            rep.scenarios.pivot_table(index="company", columns="shockBps", values="coverageFull")
            .round(1)
            .rename(columns=lambda c: f"{c:+d} bp"),
        )
        st.caption("Stressed EBIT interest coverage (×) after full repricing.")
with tabs[3]:
    ui.table(rep.panel.drop(columns=["flagLowRate", "flagHighRate"], errors="ignore"))
    st.download_button(
        "Download company-year panel (CSV)",
        rep.panel.to_csv(index=False).encode(),
        "rate_sensitivity_panel.csv",
        "text/csv",
    )

ui.report_footer(
    rep.methodology, rep.caveats, [rep.benchmark.source, "FMP income statements & balance sheets (annual)"]
)
ui.download_report(b.rate_sensitivity_html(rep), f"rate_sensitivity_{date.today():%Y%m%d}.html")
