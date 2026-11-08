import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd
import streamlit as st

from src.backtest.data import BacktestData, load_backtest_data
from src.backtest.lab import LabReport, run_lab
from src.backtest.simulate import Simulation, simulate
from src.backtest.strategies import NON_TRIALS, STRATEGIES
from src.config import BACKTEST_BENCHMARK, BACKTEST_COST_BPS, BACKTEST_TOP_N
from src.dashboard import ui
from src.reporting import charts
from src.reporting.html_report import Kpi

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


def _compare_tab() -> None:
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
        return

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
                pl.style.format({"alphaAnnual": "{:+.1%}", "alphaP": "{:.3f}"}),
                width="stretch",
                hide_index=True,
            )
    ui.report_footer(
        ev.methodology,
        ev.caveats + rep.data_notes,
        [
            "FMP daily dividend-adjusted prices and annual statements (local Parquet store)",
            "FRED TB3MS (3-month T-bill)",
        ],
    )


_REBAL = {"Monthly": 1, "Quarterly": 3, "Semi-annually": 6, "Annually": 12}


@st.cache_data(ttl=3600, show_spinner=False)
def _simulate(
    strategy: str, start: str, cost: float, rebalance: int, top_n: int, capital: float
) -> tuple[LabReport, Simulation]:
    rep = run_lab(
        [strategy],
        start=start,
        cost_bps=cost,
        data=_data(),
        placebo_seeds=0,
        rebalance_every=rebalance,
        top_n=None if STRATEGIES[strategy].hold_all else top_n,
        keep_weights=True,
    )
    bench = BACKTEST_BENCHMARK.lower()
    return rep, simulate(rep.results[strategy], _data(), rep.results[bench], capital)


def _simulator_tab() -> None:
    st.markdown(
        "Simulate a real-money portfolio run by one strategy — by default the app's own **fundamental "
        "scorecard** (quality, growth, financial health and trend; the same bands and weights as the Research "
        "Report), rebuilt point-in-time from filed annual statements. See what it held, why, what it traded, "
        "and how it compared with putting the same money in SPY."
    )
    choices = [n for n in STRATEGIES if n not in NON_TRIALS and n != BACKTEST_BENCHMARK.lower()]
    with st.form("sim"):
        c1, c2, c3 = st.columns([4, 2, 2])
        strategy = c1.selectbox(
            "Strategy",
            choices,
            index=choices.index("scorecard_top") if "scorecard_top" in choices else 0,
            format_func=lambda n: f"{n} — {STRATEGIES[n].description}",
        )
        capital = c2.number_input("Starting capital ($)", 1_000.0, 1e9, 100_000.0, 10_000.0)
        rebalance = c3.selectbox("Rebalance", list(_REBAL), index=1)
        c4, c5, c6 = st.columns([2, 2, 2])
        top_n = int(
            c4.number_input(
                "Positions (top N)",
                5,
                200,
                int(BACKTEST_TOP_N),
                5,
                help="Ignored by rules that hold every qualifying stock (scorecard_buy).",
            )
        )
        start = c5.text_input("Start (month-end)", value="2011-01-31", key="sim_start")
        cost = c6.number_input(
            "Cost (bps per unit turnover)", 0.0, 100.0, float(BACKTEST_COST_BPS), 1.0, key="sim_cost"
        )
        go_sim = st.form_submit_button("Simulate portfolio", type="primary")
    if go_sim:
        try:
            with st.spinner("Simulating point-in-time portfolio…"):
                st.session_state["sim_result"] = _simulate(
                    strategy, start, float(cost), _REBAL[rebalance], top_n, float(capital)
                )
        except ValueError as exc:
            st.error(str(exc))
    got: tuple[LabReport, Simulation] | None = st.session_state.get("sim_result")
    if got is None:
        ui.empty_state("Pick a strategy and click Simulate portfolio.")
        return
    rep, sim = got
    bench = BACKTEST_BENCHMARK.lower()
    row = rep.evaluation.table.loc[sim.strategy]
    brow = rep.evaluation.table.loc[bench]
    end_v, end_b = float(sim.value.iloc[-1]), float(sim.benchmark_value.iloc[-1])
    ui.kpi_row(
        [
            Kpi(
                "Alpha vs SPY",
                f"{row.get('alphaAnnual', float('nan')):+.1%}",
                f"p = {row.get('alphaP', float('nan')):.2f} (HAC) · beta {row.get('beta', float('nan')):.2f}",
            ),
            Kpi("Sharpe", f"{row['sharpe']:.2f}", f"SPY {brow['sharpe']:.2f}"),
            Kpi("Max drawdown", f"{row['maxDrawdown']:.0%}", f"SPY {brow['maxDrawdown']:.0%}"),
            Kpi("CAGR", f"{row['cagr']:.1%}", f"SPY {brow['cagr']:.1%}"),
            Kpi(
                "Final value",
                f"${end_v:,.0f}",
                f"SPY ${end_b:,.0f} · not risk-adjusted",
            ),
        ]
    )
    ui.findings(rep.evaluation.findings)
    ui.chart(charts.portfolio_value_chart(sim.value, sim.benchmark_value, t, sim.strategy, "SPY"), key="simv")

    st.markdown("##### Holdings")
    dates = sorted(sim.holdings)
    when = st.select_slider(
        "Rebalance date", options=dates, value=dates[-1], format_func=lambda d: d.strftime("%Y-%m")
    )
    h = sim.holdings[when]
    cash = max(0.0, 1.0 - float(h["weight"].sum()))
    st.caption(
        f"{len(h)} positions · cash {cash:.1%} · portfolio value ${float(sim.value.get(when, float('nan'))):,.0f}. "
        "Scores are what the scorecard saw on that date, from the latest annual statement filed by then "
        "(fiscal period shown)."
        + (
            " Valuation is a SEPARATE screen here: earnings yield had to be at or above the universe median "
            "(shown); it is not part of the composite score."
            if "earningsYield" in h.columns
            else ""
        )
    )
    fmt: dict[Any, str | Callable[[object], str] | None] = {
        "weight": "{:.1%}",
        "value": "${:,.0f}",
        "composite": "{:.0f}",
        "earningsYield": "{:.1%}",
        "universeMedianEY": "{:.1%}",
    }
    fmt.update({k: "{:.0f}" for k in ("quality", "growth", "financialHealth", "momentum") if k in h.columns})
    st.dataframe(h.style.format(fmt, na_rep="n/a"), width="stretch")
    if "businessModel" in h.columns:
        mix = h["businessModel"].value_counts()
        st.caption(
            "Business models held: " + ", ".join(f"{k} {v}" for k, v in mix.items()) + ". "
            "n/a = not scored (no data, or not applicable to that business model, e.g. insurers' financial health)."
        )
    tr = sim.trades[sim.trades["date"] == when].drop(columns="date")
    with st.expander(f"Trades on {when:%Y-%m} ({len(tr)})"):
        st.dataframe(
            tr.style.format({"weightBefore": "{:.1%}", "weightAfter": "{:.1%}", "amount": "${:+,.0f}"}),
            width="stretch",
            hide_index=True,
        )
    ui.chart(charts.allocation_area_chart(sim.sectors, t, "Sector allocation"), key="simsec")

    st.markdown("##### Calendar-year returns")
    st.dataframe(
        sim.annual.style.format(
            {"strategy": "{:+.1%}", "benchmark": "{:+.1%}", "difference": "{:+.1%}", "months": "{:.0f}"}
        ),
        width="stretch",
    )
    ui.report_footer(
        rep.evaluation.methodology,
        rep.evaluation.caveats + sim.notes + rep.data_notes,
        [
            "FMP daily prices (close + dividend-adjusted) and annual statements (local Parquet store)",
            "FRED TB3MS (3-month T-bill)",
        ],
    )


tab_cmp, tab_sim = st.tabs(["Compare strategies", "Portfolio simulator"])
with tab_cmp:
    _compare_tab()
with tab_sim:
    _simulator_tab()
