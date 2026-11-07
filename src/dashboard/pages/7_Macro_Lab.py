import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import streamlit as st

from src.analysis import timeseries as ts
from src.analysis.macro_lab import MacroLabReport, run_macro_lab
from src.config import MACRO_SERIES_CATALOG
from src.dashboard import ui
from src.reporting import builders as b

t = ui.setup_page("Macro Lab", "🧪")
ui.page_header(
    "Quant Lab",
    "Macro Lab — multi-variable analysis",
    "Combine any macro series, prices or sector/industry P/E histories. The engine tests stationarity, then "
    "measures correlation, lead/lag, multivariate drivers, business cycles and a back-tested forecast — every "
    "result with its statistical significance and methodology.",
)

PRESETS = {
    "China rates & inflation": ["cn_interbank_3m", "cn_discount_rate", "cn_cpi_yoy", "usd_cny"],
    "US rates & auto demand": ["us_fed_funds", "us_auto_loan_rate", "us_auto_sales", "us_consumer_credit"],
    "China vs US policy": ["cn_interbank_3m", "us_fed_funds", "us_10y", "usd_cny"],
    "US cycle": ["us_industrial_production", "us_unemployment", "us_curve_10y2y", "us_fed_funds"],
    "Auto industry P/E vs rates": ["industrype:Auto - Manufacturers", "us_10y", "us_auto_sales"],
}
FREQ_LABELS = {"auto": "Auto (coarsest)", "M": "Monthly", "Q": "Quarterly", "A": "Annual"}


def _label(key: str) -> str:
    meta = MACRO_SERIES_CATALOG.get(key)
    return f"{meta['label']}  ·  {meta['region']}" if meta else key


@st.cache_data(ttl=3600, show_spinner=False)
def _run(
    keys: tuple[str, ...],
    freq: str,
    transform: str,
    start: str,
    target: str,
    horizon: int,
    method: str,
    max_lag: int,
) -> MacroLabReport:
    return run_macro_lab(
        list(keys),
        freq=freq,
        transform=transform,
        start=start,
        target=target,
        horizon=horizon,
        method=method,
        max_lag=max_lag,
    )


preset = st.pills("Presets", list(PRESETS), selection_mode="single", default="China rates & inflation")
default_keys = PRESETS.get(preset or "", PRESETS["China rates & inflation"])

with st.form("macro_lab"):
    c1, c2 = st.columns([3, 2])
    with c1:
        catalog_keys = [k for k in default_keys if k in MACRO_SERIES_CATALOG]
        keys = st.multiselect("Series", list(MACRO_SERIES_CATALOG), default=catalog_keys, format_func=_label)
        extra = st.text_input(
            "Add any series (comma-separated)",
            value=", ".join(k for k in default_keys if k not in MACRO_SERIES_CATALOG),
            placeholder="fred:DGS2, worldbank:IND:FR.INR.LEND, price:TSLA, industrype:Auto - Manufacturers",
            help="fred:<ID> · worldbank:<ISO3>:<indicator> · price:<ticker> · sectorpe:<sector> · industrype:<industry>",
        )
    with c2:
        freq = st.segmented_control(
            "Frequency", list(FREQ_LABELS), default="auto", format_func=lambda f: FREQ_LABELS[f]
        )
        transform = st.selectbox(
            "Stationary transform for inference",
            list(ts.STATIONARY_TRANSFORMS),
            index=list(ts.STATIONARY_TRANSFORMS).index("yoy_diff"),
            format_func=lambda k: ts.TRANSFORMS[k],
            help="Correlations on trending levels are often spurious. Use changes (Δ) for rates, % changes for levels.",
        )
        start = st.date_input("Start", value=date(2010, 1, 1), min_value=date(1950, 1, 1))
    all_keys = keys + [x.strip() for x in extra.split(",") if x.strip()]
    c3, c4, c5, c6 = st.columns(4)
    target = c3.selectbox("Target (dependent) variable", all_keys or ["—"], format_func=_label)
    horizon = c4.slider("Forecast horizon (periods)", 3, 36, 12)
    max_lag = c5.slider("Max lead/lag (periods)", 3, 36, 12)
    method = c6.radio("Correlation", ["pearson", "spearman"], horizontal=True, format_func=str.title)
    submitted = st.form_submit_button("Run analysis", type="primary")

if submitted:
    if not all_keys:
        st.error("Select at least one series.")
    else:
        try:
            with st.spinner("Loading series and running statistics…"):
                st.session_state["macro_lab_result"] = _run(
                    tuple(all_keys),
                    freq or "auto",
                    transform,
                    start.isoformat(),
                    target or "",
                    horizon,
                    method,
                    max_lag,
                )
        except ValueError as exc:
            st.error(str(exc))

rep: MacroLabReport | None = st.session_state.get("macro_lab_result")
if rep is None:
    ui.empty_state("Pick a preset or your own series, then Run analysis.")
    st.stop()

ui.kpi_row(b.macro_kpis(rep))
ui.findings([f.text for f in rep.findings])

view_options = {"levels": "Levels", "zscore": "Z-score", "index": "Index = 100"}
tabs = st.tabs(["Overview", "Correlation", "Lead / lag", "Regression", "Cycles", "Forecast", "Data"])

with tabs[0]:
    units = set(b.macro_units(rep).values())
    allowed = list(view_options) if len(units) == 1 else ["zscore", "index"]
    view = st.segmented_control(
        "Scale",
        allowed,
        default=b.macro_default_view(rep),
        format_func=lambda v: view_options[v],
        key="ml_view",
    )
    figs = b.macro_figures(rep, t, view=view or b.macro_default_view(rep))
    if len(units) > 1:
        st.caption(
            "Series have different units, so the shared chart is standardised; see small multiples for raw values."
        )
    ui.chart(figs["overview"])
    ui.chart(figs["multiples"])
    st.markdown("##### Descriptive statistics")
    ui.table(b.macro_stats_table(rep))
    st.caption(
        "Trend / yr: OLS slope per year (* = significant at 5%). ADF: Augmented Dickey-Fuller, 5% critical value −2.86."
    )

with tabs[1]:
    if rep.levels.shape[1] < 2:
        st.info("Add a second series to compute correlations.")
    else:
        st.markdown(
            f"Inference basis: **{rep.recommended_basis}** "
            f"({'levels are non-stationary — use the transformed view' if rep.recommended_basis == 'transformed' else 'levels are stationary'})."
        )
        c1, c2 = st.columns(2)
        with c1:
            if "corr_transformed" in figs:
                ui.chart(figs["corr_transformed"])
        with c2:
            if "corr_levels" in figs:
                ui.chart(figs["corr_levels"])
        st.markdown("##### Stability over time")
        cols = list(rep.levels.columns)
        a_col, b_col, w_col = st.columns([2, 2, 1])
        a = a_col.selectbox("Series A", cols, index=0)
        bb = b_col.selectbox("Series B", cols, index=1 if len(cols) > 1 else 0)
        window = w_col.number_input("Window", min_value=6, max_value=120, value=24)
        basis = rep.transformed if rep.recommended_basis == "transformed" else rep.levels
        if a != bb and len(basis) > window:
            from src.reporting import charts

            ui.chart(
                charts.rolling_corr_chart(
                    ts.rolling_correlation(basis, a, bb, int(window)),
                    t,
                    a,
                    bb,
                    int(window),
                    b.FREQ_UNIT[rep.freq],
                )
            )

with tabs[2]:
    if not rep.lead_lag:
        st.info("Lead/lag needs a target plus at least one driver.")
    for d, fig in figs["lead_lag"].items():
        ui.chart(fig, key=f"ll_{d}")

with tabs[3]:
    reg = b.regression_table(rep)
    if reg is None:
        st.info("Regression needs a target and at least one driver with enough overlapping data.")
    else:
        r = rep.regression or {}
        st.markdown(
            f"**{rep.target}** on all other series · R² **{r['r2']:.2f}** (adj. {r['adjR2']:.2f}) · n = {r['n']} · "
            f"Durbin-Watson {r['durbinWatson']:.2f} · HAC lags {r['hacLags']}"
        )
        ui.table(reg)
        ui.chart(figs["fit"])

with tabs[4]:
    if not rep.cycles:
        st.info("Not enough history for cycle analysis.")
    else:
        ui.table(b.cycle_summary_table(rep))
        pick = st.selectbox(
            "Series",
            list(rep.cycles),
            index=list(rep.cycles).index(rep.target) if rep.target in rep.cycles else 0,
        )
        ui.chart(figs["cycles"][pick], key=f"cyc_{pick}")
        ui.chart(figs["spectra"][pick], key=f"spec_{pick}")
        st.caption(
            "'Indicative next turn' = last turning point + half the average cycle. A timing heuristic, not a forecast."
        )

with tabs[5]:
    if "forecast" not in figs:
        st.info("Forecast not available at this frequency / history length.")
    else:
        ui.chart(figs["forecast"])
        fc = rep.forecast or {}
        st.markdown(
            f"Back-test on the last **{fc['holdoutPeriods']}** periods: RMSE model **{fc['rmseModel']:.3f}** vs naive "
            f"**{fc['rmseNaive']:.3f}** → skill **{fc['skill']:+.0%}**."
        )

with tabs[6]:
    st.markdown("##### Aligned levels")
    ui.table(rep.levels)
    st.download_button(
        "Download aligned data (CSV)", rep.levels.to_csv().encode(), "macro_lab_levels.csv", "text/csv"
    )
    st.markdown(f"##### Transformed — {ts.TRANSFORMS[rep.transform]}")
    ui.table(rep.transformed)

ui.report_footer(rep.methodology, rep.caveats, rep.sources)
ui.download_report(b.macro_lab_html(rep), f"macro_lab_{date.today():%Y%m%d}.html")
