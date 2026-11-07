import html
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import streamlit as st

from src.analysis.indicators import (
    GROUPS,
    INDICATORS,
    CompanyData,
    get_profile,
    indicators_by_group,
    load_company,
    load_prices,
    profile_defaults,
    trend_findings,
)
from src.config import INDICATOR_HISTORY_YEARS, MARKET_BENCHMARK_SYMBOL
from src.dashboard import ui
from src.data.fmp_client import FMPClient, FMPError
from src.reporting import builders as b

t = ui.setup_page("Stock Analysis", "🔎")
ui.page_header(
    "Research",
    "Stock analysis — fundamentals over time",
    "Up to 10 fiscal years of return, risk and valuation indicators (incl. **gearing**, net debt ÷ EBITDA, "
    "interest coverage, ROIC, FCF margin), the financial statements behind them, and price return / risk. "
    "Indicator defaults adapt to the industry (banks, REITs, software, ...).",
)


@st.cache_data(ttl=3600, show_spinner=False)
def _load(symbol: str, years: int) -> CompanyData:
    client = FMPClient()
    bench = load_prices(MARKET_BENCHMARK_SYMBOL, client)
    return load_company(symbol, client, years=years, bench_prices=bench)


if "symbol" not in st.session_state:
    st.session_state["symbol"] = "AAPL"

with st.form("stock"):
    c1, c2 = st.columns([3, 2])
    sym = c1.text_input("Ticker", key="symbol")
    years = c2.slider("Fiscal years of history", 3, INDICATOR_HISTORY_YEARS, INDICATOR_HISTORY_YEARS)
    submitted = st.form_submit_button("Analyze", type="primary")

if submitted:
    try:
        with st.spinner(f"Loading {years} years of fundamentals for {sym.strip().upper()}…"):
            st.session_state["stock_ts_result"] = _load(sym.strip().upper(), years)
    except (FMPError, ValueError) as exc:
        st.error(str(exc))

c: CompanyData | None = st.session_state.get("stock_ts_result")
if c is None:
    ui.empty_state("Enter a ticker and click Analyze.")
    st.stop()

profile_key, profile = get_profile(c.sector, c.industry)
st.markdown(
    "".join(
        f"<span class='rp-tag'>{html.escape(str(x))}</span>"
        for x in (c.sector, c.industry, c.exchange, f"{profile['label']} profile")
    )
    + f"<b>{html.escape(c.company)} ({html.escape(c.symbol)})</b>",
    unsafe_allow_html=True,
)
if c.currency_mismatch:
    st.warning(
        f"Currency mismatch: {c.symbol} quotes in {c.quote_currency} but reports in {c.reporting_currency}. "
        "Valuation multiples are unavailable (no FX conversion). Margins, returns and leverage are valid; "
        f"statement figures are shown in {c.reporting_currency}."
    )
ui.kpi_row(b.stock_kpis(c))

# ------------------------------------------------------------------ indicator picker
defaults = profile_defaults(profile)
not_meaningful = set(profile.get("not_meaningful", []))
with st.expander("Indicators — choose what to show", expanded=False):
    st.caption(
        f"Defaults follow the **{profile['label']}** profile. Price-based return/risk (volatility, beta, "
        "drawdown) is on the Price tab."
    )
    cols = st.columns(3)
    selected: dict[str, list[str]] = {}
    for col, (g, label) in zip(cols, GROUPS.items(), strict=True):
        opts = [i.key for i in indicators_by_group(g) if not i.market]
        selected[g] = col.multiselect(
            label,
            opts,
            default=[k for k in defaults[g] if k in opts],
            format_func=lambda k: f"{INDICATORS[k].label} · {INDICATORS[k].category}",
            key=f"sa_{g}_{profile_key}",
        )
keys = [k for g in GROUPS for k in selected[g]]
flagged = [INDICATORS[k].label for k in keys if k in not_meaningful]
if flagged:
    st.info(f"Not meaningful for {profile['label']}: {', '.join(flagged)} — interpret with care.")

ui.findings(trend_findings(c, keys))

figs = b.stock_figures(c, keys, t)
tabs = st.tabs(["Return", "Risk", "Valuation", "Financial statements", "Price & risk", "Data"])
for tab, g in zip(tabs[:3], GROUPS, strict=True):
    with tab:
        if g not in figs:
            st.info("No indicators selected for this group — open the Indicators panel above.")
            continue
        if g == "valuation" and c.currency_mismatch:
            st.info("Valuation multiples are blank because of the currency mismatch.")
        ui.chart(figs[g], key=f"sa_fig_{g}")
        ui.table(b.indicator_history_table(c, selected[g]).drop(columns="Group", errors="ignore"))
        with st.expander("Definitions"):
            ui.bullet_card([f"**{INDICATORS[k].label}** — {INDICATORS[k].description}" for k in selected[g]])
with tabs[3]:
    ui.chart(figs["statements"], key="sa_fig_statements")
    st.caption(f"Money in {c.reporting_currency} (reporting currency).")
    ui.table(b.statement_table(c))
with tabs[4]:
    if "price" in figs:
        ui.chart(figs["price"], key="sa_fig_price")
        ui.chart(figs["drawdown"], key="sa_fig_dd")
        ui.table(b.market_table(c))
    else:
        st.info("Price history unavailable.")
with tabs[5]:
    ui.table(b.indicator_history_table(c, keys))
    st.download_button(
        "Download indicator history (CSV)",
        c.values.to_csv().encode(),
        f"{c.symbol}_indicators.csv",
        "text/csv",
    )

ui.report_footer(
    [
        "Each indicator is computed per fiscal year from FMP annual statements, ratios and key metrics "
        "(Starter tier: up to 10 years — quarterly ratios need FMP Premium).",
        "Balance-sheet ratios use period-end values; ROE, gearing and debt ÷ equity are blank when equity <= 0; "
        "growth is blank when the prior-year base is <= 0.",
        "Time-series valuation uses the fiscal-year-end market cap; the P/E tile uses TTM.",
        f"Price metrics: daily closes (dividends excluded); beta vs {MARKET_BENCHMARK_SYMBOL} from weekly log "
        "returns, Newey-West HAC standard errors.",
    ],
    b.stock_caveats(c),
    [
        "FMP income statement, balance sheet, cash flow, ratios, key metrics (annual + TTM), profile, daily prices"
    ],
)
ui.download_report(
    b.stock_analysis_html(c, keys, trend_findings(c, keys)),
    f"{c.symbol}_fundamentals_{date.today():%Y%m%d}.html",
)
