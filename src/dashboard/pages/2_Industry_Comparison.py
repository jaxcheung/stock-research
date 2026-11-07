import html
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd
import streamlit as st

from src.analysis.indicators import GROUPS, INDICATORS, indicators_by_group
from src.analysis.industry_analysis import (
    SCOPES,
    SIZE_MODES,
    IndustryComparisonReport,
    PeerFilters,
    comparison_findings,
    price_table,
    run_industry_comparison,
)
from src.config import (
    CAP_SIZE_BUCKETS,
    INDICATOR_HISTORY_YEARS,
    INDUSTRY_COMP_MAX_PEERS,
    INDUSTRY_COMP_MAX_PEERS_LIMIT,
)
from src.dashboard import ui
from src.data.fmp_client import FMPError
from src.reporting import builders as b
from src.reporting.formatting import value

t = ui.setup_page("Industry Comparison", "🏭")
ui.page_header(
    "Research",
    "Industry comparison — return, risk and valuation over time",
    "A company against a filtered peer group: **return** (growth, margins, ROE/ROIC, price return) and "
    "**risk** (gearing, coverage, liquidity, volatility, beta) compared separately, as 10-year time series "
    "and today's snapshot, with industry aggregates (median and Σ÷Σ, e.g. industry P/E = Σ market cap ÷ "
    "Σ earnings). Indicators adapt to the industry.",
)


def _tickers(text: str) -> tuple[str, ...]:
    return tuple(s.strip().upper() for s in text.replace(";", ",").replace(" ", ",").split(",") if s.strip())


@st.cache_data(ttl=3600, show_spinner=False)
def _run(
    symbol: str,
    scope: str,
    size_mode: str,
    buckets: tuple[str, ...],
    us_only: bool,
    max_peers: int,
    include: tuple[str, ...],
    exclude: tuple[str, ...],
) -> IndustryComparisonReport:
    f = PeerFilters(scope, size_mode, buckets, us_only, max_peers, include, exclude)
    return run_industry_comparison(symbol, f, years=INDICATOR_HISTORY_YEARS)


if "symbol" not in st.session_state:
    st.session_state["symbol"] = "AAPL"

with st.form("industry"):
    r1 = st.columns([2, 2, 2, 2])
    sym = r1[0].text_input("Ticker", key="symbol")
    scope = r1[1].selectbox(
        "Peer scope",
        list(SCOPES),
        format_func=lambda k: SCOPES[k],
        help="One scope at a time — industry and sector peer sets are never blended.",
    )
    max_peers = r1[2].slider("Max peers", 3, INDUSTRY_COMP_MAX_PEERS_LIMIT, INDUSTRY_COMP_MAX_PEERS)
    us_only = r1[3].toggle(
        "US-domiciled only",
        value=False,
        help="Drops foreign ADRs (their valuation is blank anyway when they report in another currency).",
    )
    r2 = st.columns([2, 3, 3])
    size_mode = r2[0].radio("Size filter", list(SIZE_MODES), format_func=lambda k: SIZE_MODES[k], index=1)
    buckets = r2[1].multiselect(
        "Cap-size buckets (when 'Cap-size buckets' is chosen)",
        list(CAP_SIZE_BUCKETS),
        default=["Large ($10B–200B)", "Mega (≥ $200B)"],
    )
    with r2[2]:
        include = st.text_input("Always include tickers", placeholder="e.g. MSFT, GOOGL")
        exclude = st.text_input("Exclude tickers", placeholder="e.g. BRK-A")
    st.caption(
        "'Similar to subject' = 0.2×–5× its market cap. First run of a 10–25 company group takes ~15–60 s "
        "(≈7 FMP calls per company), then it is cached."
    )
    submitted = st.form_submit_button("Compare", type="primary")

if submitted:
    try:
        with st.spinner(f"Building peer group and 10-year panel for {sym.strip().upper()}…"):
            st.session_state["industry_ts_result"] = _run(
                sym.strip().upper(),
                scope or "industry",
                size_mode or "similar",
                tuple(buckets),
                us_only,
                max_peers,
                _tickers(include),
                _tickers(exclude),
            )
    except (FMPError, ValueError) as exc:
        st.error(str(exc))

rep: IndustryComparisonReport | None = st.session_state.get("industry_ts_result")
if rep is None:
    ui.empty_state("Choose a ticker and peer filters, then click Compare.")
    st.stop()

s = rep.subject
st.markdown(
    "".join(
        f"<span class='rp-tag'>{html.escape(str(x))}</span>"
        for x in (s.sector, s.industry, SCOPES[rep.filters.scope], f"{rep.profile['label']} profile")
    )
    + f"<b>{html.escape(s.company)} ({html.escape(rep.symbol)})</b>",
    unsafe_allow_html=True,
)

if rep.notice:
    st.warning(rep.notice)
if not rep.peers:
    st.warning("No peers matched these filters — widen the size filter, switch scope, or add tickers.")

# ------------------------------------------------------------------ display controls (no refetch)
defaults = rep.defaults
with st.expander("Indicators & display — choose what to compare", expanded=False):
    st.caption(
        f"Defaults follow the **{rep.profile['label']}** profile; add or remove any indicator. "
        "Changing these never re-downloads data."
    )
    cols = st.columns(3)
    selected: dict[str, list[str]] = {}
    for col, (g, label) in zip(cols, GROUPS.items(), strict=True):
        opts = [i.key for i in indicators_by_group(g)]
        selected[g] = col.multiselect(
            label,
            opts,
            default=[k for k in defaults[g] if k in opts],
            format_func=lambda k: f"{INDICATORS[k].label} · {INDICATORS[k].category}",
            key=f"ic_{g}_{rep.profile_key}",
        )
    d1, d2, d3 = st.columns(3)
    highlight = d1.multiselect(
        "Highlight peers in colour (max 5)", rep.peers, max_selections=5, key="ic_highlight"
    )
    basis_label = d2.radio(
        "Valuation snapshot basis", ["Current (TTM)", "Latest fiscal year"], horizontal=True, key="ic_basis"
    )
    last_n = d3.slider("Years shown in time series", 3, INDICATOR_HISTORY_YEARS, INDICATOR_HISTORY_YEARS)
basis = "ttm" if basis_label == "Current (TTM)" else "latest"
keys = [k for g in GROUPS for k in selected[g]]
flagged = [INDICATORS[k].label for k in keys if k in rep.not_meaningful]
if flagged:
    st.info(f"Not meaningful for {rep.profile['label']}: {', '.join(flagged)} — interpret with care.")

ui.kpi_row(b.industry_kpis(rep, basis))
findings = comparison_findings(rep, keys, basis)
ui.findings(findings)

figs = b.industry_figures(rep, keys, t, highlight, basis, last_n)
tabs = st.tabs(["Return", "Risk", "Valuation", "Market return & risk", "Peer universe", "Data"])
for tab, g in zip(tabs[:3], GROUPS, strict=True):
    with tab:
        gk = selected[g]
        if not gk:
            st.info("No indicators selected for this group — open 'Indicators & display' above.")
            continue
        st.markdown(f"##### {GROUPS[g]} — latest snapshot")
        ui.table(b.comparison_table(rep, gk, basis))
        st.caption(
            "★ = subject. Peer statistics exclude the subject. Mean is not shown for multiples (averaging ratios "
            "is biased) — use the Σ÷Σ aggregate. "
            + (
                "Valuation snapshot: TTM where FMP provides it."
                if g == "valuation" and basis == "ttm"
                else ""
            )
        )
        if f"ts_{g}" in figs:
            ui.chart(figs[f"ts_{g}"], key=f"ic_ts_{g}")
        else:
            st.caption("Price-based indicators have no fiscal-year time series — see 'Market return & risk'.")
        rank_key = st.selectbox(
            "Rank peers by",
            gk,
            format_func=lambda k: INDICATORS[k].label,
            key=f"ic_rank_{g}",
        )
        ui.chart(figs[f"rank_{rank_key}"], key=f"ic_rank_fig_{g}")
with tabs[3]:
    if "cumret" in figs:
        ui.chart(figs["cumret"], key="ic_cumret")
        ui.chart(figs["scatter"], key="ic_scatter")
        ui.chart(figs["rollvol"], key="ic_rollvol")
        ui.chart(figs["drawdown"], key="ic_dd")
        pt = price_table(rep)
        fmt = {"ret1y": "pct", "cagr3y": "pct", "cagr5y": "pct", "volatility": "pct", "maxDrawdown": "pct"}
        shown = pd.DataFrame(
            {INDICATORS[k].label: [value(v, f) for v in pt[k]] for k, f in fmt.items()}, index=pt.index
        )
        shown["Beta vs market"] = [
            f"{bv:.2f} (p={pv:.3f}, n={int(nv)})" if pd.notna(bv) else "—"
            for bv, pv, nv in zip(pt["beta"], pt["betaP"], pt["betaN"], strict=True)
        ]
        ui.table(shown)
    else:
        st.info("Price history unavailable.")
with tabs[4]:
    ui.table(b.peers_display_table(rep))
    if rep.skipped:
        st.caption("Skipped: " + "; ".join(f"{sy} — {r[:80]}" for sy, r in rep.skipped))
with tabs[5]:
    long = pd.concat(
        {sy: c.values for sy, c in rep.companies.items()}, names=["symbol", "fiscalYear"]
    ).reset_index()
    ui.table(long.round(4), hide_index=True)
    st.download_button(
        "Download indicator panel (CSV)",
        long.to_csv(index=False).encode(),
        f"{rep.symbol}_peer_panel.csv",
        "text/csv",
    )

ui.report_footer(
    rep.methodology,
    rep.caveats,
    [
        "FMP company screener / stock-peers (universe)",
        "FMP statements, ratios, key metrics (annual + TTM), profile",
        "FMP daily price history (incl. SPY benchmark)",
    ],
)
ui.download_report(
    b.industry_comparison_html(rep, keys, highlight, findings, basis, last_n),
    f"{rep.symbol}_industry_comparison_{date.today():%Y%m%d}.html",
)
