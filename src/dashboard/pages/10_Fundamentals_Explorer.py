import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import duckdb
import streamlit as st

from src.analysis.fundamentals_panel import (
    METRICS,
    PanelReport,
    analyze_panel,
    growth_leaderboard,
    sector_trend,
)
from src.config import SYNC_DEFAULT_YEARS
from src.dashboard import ui
from src.data import store
from src.data.fmp_client import FMPError
from src.data.sync import build_universe, sync_statements
from src.reporting import builders as b
from src.reporting import charts

t = ui.setup_page("Fundamentals Explorer", "🗄️")
ui.page_header(
    "Quant Lab",
    "Fundamentals Explorer — 10-year panel",
    "Ten years of income statements for hundreds of companies, stored locally as Parquet and queried with "
    "DuckDB. Sector margin and growth trends, a growth-vs-profitability leaderboard, company drill-downs and a "
    "read-only SQL workbench.",
)


@st.cache_data(ttl=600, show_spinner=False)
def _analyze(stamp: float) -> PanelReport:
    return analyze_panel()


def _store_stamp() -> float:
    folder = store.STORE_DIR / "fmp" / "statements" / "income_statement" / "annual"
    files = list(folder.glob("*.parquet")) if folder.exists() else []
    return max((f.stat().st_mtime for f in files), default=0.0) + len(files)


tabs = st.tabs(["Sector trends", "Leaderboard", "Company", "SQL workbench", "Data store"])

# Data store tab is built first so a sync can run even when the store is empty.
with tabs[4]:
    db_status = store.StoreDB()
    st.markdown("##### Store contents")
    if db_status.tables:
        ui.table(db_status.describe())
    else:
        st.info("The store is empty — run a sync below.")
    st.caption(
        f"Location: `{store.STORE_DIR}` (gitignored). Universes: {', '.join(store.list_universes()) or 'none'}."
    )
    db_status.close()
    with st.form("sync"):
        st.markdown("##### Download / refresh statements")
        c1, c2, c3, c4 = st.columns(4)
        uni = c1.text_input("Universe name", "us_large")
        min_cap = c2.number_input("Min market cap ($bn)", min_value=0.5, value=10.0, step=0.5)
        country = c3.text_input("Country (ISO2, blank = any)", "US")
        years = c4.slider("Years", 1, 10, SYNC_DEFAULT_YEARS)
        kinds = st.multiselect("Statements", list(store.STATEMENT_KINDS), default=["income_statement"])
        extra = st.text_input("Or explicit tickers (comma-separated, overrides screen)", "")
        go_sync = st.form_submit_button("Run sync", type="primary")
    if go_sync:
        try:
            if extra.strip():
                syms = [s.strip().upper() for s in extra.split(",") if s.strip()]
                store.save_universe(uni, [{"symbol": s} for s in syms])
            else:
                syms = [r["symbol"] for r in build_universe(uni, min_cap * 1e9, country.strip() or None)]
            bar = st.progress(0.0, text="Starting…")

            def _progress(i: int, n: int, s: str) -> None:
                bar.progress(i / max(n, 1), text=f"{i}/{n} {s}")

            rep_sync = sync_statements(
                syms,
                uni,
                tuple(kinds),
                "annual",
                years,
                progress=_progress,
            )
            st.success(rep_sync.summary())
            if rep_sync.failed:
                st.warning(
                    "Failed: " + ", ".join(f"{k} ({v[:60]})" for k, v in list(rep_sync.failed.items())[:20])
                )
            st.cache_data.clear()
        except (FMPError, ValueError) as exc:
            st.error(str(exc))

try:
    rep = _analyze(_store_stamp())
except (ValueError, duckdb.Error) as exc:
    for tab in tabs[:4]:
        with tab:
            ui.empty_state(f"{exc} Use the Data store tab to download statements.")
    st.stop()

yrs = rep.complete_years

with tabs[0]:
    ui.kpi_row(b.panel_kpis(rep))
    ui.findings(rep.findings)
    c1, c2 = st.columns([2, 3])
    metric = c1.selectbox(
        "Metric",
        list(METRICS),
        index=list(METRICS).index("operatingMargin"),
        format_func=lambda m: METRICS[m]["label"],
    )
    meta = METRICS[metric]
    use_years = yrs[1:] if meta["kind"] == "change" else yrs
    med, cnt = sector_trend(rep.panel, metric, use_years)
    sectors = [s for s in med.index if s != "All companies"]
    highlight = c2.multiselect("Highlight sectors (max 6)", sectors, default=sectors[:3], max_selections=6)
    ui.chart(charts.sector_heatmap(med, cnt, t, meta["label"], meta["fmt"], meta["kind"] == "change"))
    ui.chart(charts.sector_lines(med, t, meta["label"], meta["fmt"], highlight))

with tabs[1]:
    if len(yrs) < 2:
        st.info("Need at least two complete years.")
    else:
        c1, c2, c3 = st.columns(3)
        y0 = c1.selectbox("From", yrs, index=0)
        y1 = c2.selectbox("To", [y for y in yrs if y > y0], index=len([y for y in yrs if y > y0]) - 1)
        sec = c3.selectbox("Highlight sector", ["(none)", *sorted(rep.panel["sector"].dropna().unique())])
        lb = growth_leaderboard(rep.panel, y0, y1)
        ui.chart(charts.leaderboard_scatter(lb, t, None if sec == "(none)" else sec))
        view = lb if sec == "(none)" else lb[lb["sector"] == sec]
        ui.table(b.leaderboard_table(view, top=100))
        st.download_button(
            "Download leaderboard (CSV)",
            lb.to_csv(index=False).encode(),
            f"leaderboard_{y0}_{y1}.csv",
            "text/csv",
        )

with tabs[2]:
    uniq = rep.panel.drop_duplicates("symbol")
    names = {str(s): str(c) for s, c in zip(uniq["symbol"], uniq["company"].fillna(""), strict=True)}
    syms = sorted(names)
    default = st.session_state.get("symbol", "AAPL")
    sym = st.selectbox(
        "Company",
        syms,
        index=syms.index(default) if default in syms else 0,
        format_func=lambda s: f"{s} — {names.get(s, '')}",
    )
    hist = rep.panel[rep.panel["symbol"] == sym].sort_values("year")
    ccy = str(hist["currency"].iloc[-1]) if not hist.empty else ""
    c1, c2 = st.columns(2)
    with c1:
        ui.chart(charts.company_financials(hist, t, sym, ccy))
    with c2:
        ui.chart(charts.company_margins(hist, t, sym))
    cols = [
        "year",
        "revenue",
        "netIncome",
        "epsDiluted",
        "grossMargin",
        "operatingMargin",
        "netMargin",
        "revenueGrowth",
        "rdIntensity",
        "interestBurden",
        "effectiveTaxRate",
    ]
    ui.table(hist[cols].set_index("year").round(4))

with tabs[3]:
    st.markdown(
        "Read-only DuckDB over the local store. Tables: `income_annual`, `companies`, `series` "
        "(plus `balance_annual` / `cashflow_annual` if synced). Single SELECT only; 5,000-row cap; no file access."
    )
    examples = {
        "Median net margin by sector, latest year": (
            "SELECT c.sector, year(CAST(i.date AS DATE)) AS yr, count(*) AS n,\n"
            "       median(i.netIncome / i.revenue) AS median_net_margin\n"
            "FROM income_annual i JOIN companies c USING (symbol)\n"
            "WHERE i.revenue > 0 AND year(CAST(i.date AS DATE)) = 2025\n"
            "GROUP BY 1, 2 ORDER BY median_net_margin DESC"
        ),
        "Companies with 10 straight years of revenue growth": (
            "WITH g AS (\n  SELECT symbol, revenue > lag(revenue) OVER (PARTITION BY symbol ORDER BY date) AS up\n"
            "  FROM income_annual)\nSELECT symbol, count(*) FILTER (WHERE up) AS growth_years\n"
            "FROM g GROUP BY symbol HAVING growth_years >= 9 ORDER BY growth_years DESC"
        ),
        "Largest R&D spenders (latest year, USD reporters)": (
            "SELECT symbol, date, researchAndDevelopmentExpenses / 1e9 AS rd_bn,\n"
            "       researchAndDevelopmentExpenses / revenue AS rd_intensity\n"
            "FROM income_annual WHERE reportedCurrency = 'USD'\n"
            "QUALIFY row_number() OVER (PARTITION BY symbol ORDER BY date DESC) = 1\n"
            "ORDER BY rd_bn DESC LIMIT 25"
        ),
    }
    pick = st.selectbox("Example", list(examples))
    sql = st.text_area("SQL", examples[pick], height=170)
    if st.button("Run query", type="primary"):
        ro = store.StoreDB(read_only_sql=True)
        try:
            res = ro.query_user(sql)
            st.caption(f"{len(res):,} rows")
            ui.table(res)
            st.download_button(
                "Download result (CSV)", res.to_csv(index=False).encode(), "query.csv", "text/csv"
            )
        except (ValueError, duckdb.Error) as exc:
            st.error(str(exc))
        finally:
            ro.close()

ui.report_footer(
    rep.methodology,
    rep.caveats,
    [
        "FMP annual income statements (local Parquet store, DuckDB)",
        "FMP company screener (universe snapshot)",
    ],
)
ui.download_report(b.panel_html(rep, metric, highlight), f"fundamentals_panel_{date.today():%Y%m%d}.html")
