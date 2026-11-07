import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd
import streamlit as st

from src.backtest.data import BacktestData, load_backtest_data
from src.backtest.lab import LabReport, run_lab
from src.backtest.strategies import NON_TRIALS, STRATEGIES
from src.config import BACKTEST_BENCHMARK, BACKTEST_COST_BPS
from src.dashboard import ui
from src.reporting import charts

t = ui.setup_page("Backtest Lab", "🧮")
ui.page_header(
    "Quant Lab",
    "Backtest Lab — rule-based strategies, judged honestly",
    "Point-in-time monthly backtests of factor and sector-ETF strategies on US-listed stocks ≥ $2bn, net of "
    "costs, against SPY. Overfitting controls are built in: a deflated Sharpe ratio for the number of strategies "
    "tried, bootstrap confidence intervals, a random-pick placebo and a walk-forward out-of-sample test. "
    "No LLM is involved, so the history is a clean test. Research only, not advice.",
)


@st.cache_resource(show_spinner=False)
def _data() -> BacktestData:
    return load_backtest_data()


@st.cache_data(ttl=3600, show_spinner=False)
def _run(strategies: tuple[str, ...], start: str, cost: float) -> LabReport:
    return run_lab(list(strategies), start=start, cost_bps=cost, data=_data())


trials = [n for n in STRATEGIES if n not in NON_TRIALS]
with st.form("lab"):
    c1, c2, c3 = st.columns([4, 2, 2])
    picked = c1.multiselect(
        "Strategies to test",
        trials,
        default=["multi_factor", "momentum_12_1", "quality_gp", "sector_dual_momentum"],
        format_func=lambda n: f"{n} — {STRATEGIES[n].description}",
        help="Every configuration ever run is recorded in a trial ledger and counts towards the deflated Sharpe: "
        "testing more raises the bar, and running strategies one at a time does not reset it.",
    )
    start = c2.text_input("Start (month-end)", value="2011-01-31")
    cost = c3.number_input("Cost (bps per unit turnover)", 0.0, 100.0, float(BACKTEST_COST_BPS), 1.0)
    submitted = st.form_submit_button("Run backtests", type="primary")

if submitted:
    if not picked:
        st.error("Pick at least one strategy.")
    else:
        try:
            with st.spinner("Running point-in-time backtests…"):
                st.session_state["lab_report"] = _run(tuple(picked), start, float(cost))
        except ValueError as exc:
            st.error(str(exc))

rep: LabReport | None = st.session_state.get("lab_report")
if rep is None:
    ui.empty_state(
        "Pick strategies and click Run backtests. Data comes from the local store "
        "(`python -m src.data.sync --min-cap 2e9 --kinds income_statement balance_sheet cash_flow --prices`)."
    )
    st.stop()

ev = rep.evaluation
bench = BACKTEST_BENCHMARK.lower()
ui.findings(ev.findings)

tested = [n for n in rep.results if n not in NON_TRIALS and n != bench]
colors = {n: charts.series_color(t, i) for i, n in enumerate(tested)}
curves = rep.equity_curves()
ui.chart(charts.equity_curve_chart(curves, t, bench, colors), key="eq")
ui.chart(charts.drawdown_chart(curves, t, colors, bench), key="dd")

st.markdown("##### Performance and significance")
tab = ev.table.copy()
show = pd.DataFrame(
    {
        "CAGR": tab["cagr"],
        "Volatility": tab["volatility"],
        "Sharpe": tab["sharpe"],
        "Sharpe 95% CI": tab["sharpeCI95"].map(
            lambda c: f"{c[0]:.2f} – {c[1]:.2f}" if isinstance(c, tuple) else ""
        ),
        "Max drawdown": tab["maxDrawdown"],
        "Alpha / yr": tab.get("alphaAnnual"),
        "Alpha p": tab.get("alphaP"),
        "Info ratio": tab.get("informationRatio"),
        "Deflated IR (P beats SPY)": tab.get("deflatedSharpe"),
        "Trials": tab.get("nTrials"),
        "Avg turnover": tab["avgTurnover"],
        "Months": tab["n"],
        "Start": tab["start"],
        "End": tab["end"],
        "Control": tab["control"],
    }
)
st.dataframe(
    show.style.format(
        {
            "CAGR": "{:.1%}",
            "Volatility": "{:.1%}",
            "Sharpe": "{:.2f}",
            "Max drawdown": "{:.0%}",
            "Alpha / yr": "{:+.1%}",
            "Alpha p": "{:.3f}",
            "Info ratio": "{:.2f}",
            "Deflated IR (P beats SPY)": "{:.0%}",
            "Trials": "{:.0f}",
            "Avg turnover": "{:.0%}",
        },
        na_rep="—",
    ),
    width="stretch",
)
st.caption(
    "Alpha from an excess-return regression on SPY with Newey-West (HAC) standard errors. Deflated IR = "
    "deflated Sharpe of the ACTIVE return vs SPY: the probability the strategy truly beats SPY after allowing "
    "for the best result expected by luck across every configuration ever tried (Trials; controls excluded)."
)

st.markdown("##### Walk-forward (out-of-sample)")
ui.chart(charts.walk_forward_chart(ev.walk_forward, t, bench), key="wf")
sel = ev.selection
if "best" in sel:
    p_oos = sel.get("alphaPOOS")
    st.caption(
        f"Split at {sel['splitAt']}: the best in-sample strategy was **{sel['best']}**; its out-of-sample "
        f"Sharpe was {sel['sharpeOOS']:.2f} vs SPY's {sel['benchSharpeOOS']:.2f}"
        + (
            f" (OOS alpha p = {p_oos:.2f}, n = {sel.get('nOOS')} months)."
            if isinstance(p_oos, float)
            else "."
        )
    )
pl = rep.placebo.get("table")
if isinstance(pl, pd.DataFrame) and not pl.empty:
    with st.expander(f"Placebo detail — {len(pl)} random-pick portfolios"):
        st.dataframe(
            pl.style.format({"alphaAnnual": "{:+.1%}", "alphaP": "{:.3f}"}), width="stretch", hide_index=True
        )
ui.report_footer(
    ev.methodology,
    ev.caveats + rep.data_notes,
    [
        "FMP daily dividend-adjusted prices and annual statements (local Parquet store)",
        "FRED TB3MS (3-month T-bill)",
    ],
)
