# src/reporting/builders.py
#
# Turns analysis reports into display-ready tables, figure sets and the HTML
# export. Shared by the Streamlit pages and the downloadable report so the two
# can never disagree.

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.analysis import timeseries as ts
from src.analysis.macro_lab import MacroLabReport
from src.analysis.rate_sensitivity import RateSensitivityReport
from src.analysis.relative_valuation import RelativeValuationReport
from src.config import CYCLE_MAX_PERIOD_YEARS, CYCLE_MIN_PERIOD_YEARS
from src.reporting import charts
from src.reporting.html_report import Kpi, Section, build_html_report
from src.reporting.theme import LIGHT, Theme

FREQ_UNIT = {"D": "day", "W": "week", "M": "month", "Q": "quarter", "A": "year"}


def _fmt(v: Any, spec: str = ",.2f", suffix: str = "") -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    try:
        return f"{v:{spec}}{suffix}"
    except (TypeError, ValueError):
        return str(v)


# =========================================================================== Macro Lab


def macro_colors(rep: MacroLabReport, t: Theme) -> dict[str, str]:
    # Colour follows the entity (selection order), never rank.
    return {s.label: charts.series_color(t, i) for i, s in enumerate(rep.series)}


def macro_units(rep: MacroLabReport) -> dict[str, str]:
    return {s.label: s.unit for s in rep.series}


def macro_default_view(rep: MacroLabReport) -> str:
    units = set(macro_units(rep).values())
    return "levels" if len(units) == 1 and units <= {"pct", "ratio"} else "zscore"


def macro_stats_table(rep: MacroLabReport) -> pd.DataFrame:
    s = rep.stats
    units = macro_units(rep)
    out = pd.DataFrame(index=s.index)
    out["Latest"] = [
        _fmt(v, ",.2f", charts.UNIT_SUFFIX.get(units[str(i)], ""))
        for i, v in zip(s.index, s["last"], strict=True)
    ]
    out["Period"] = s["start"].str[:7] + " → " + s["end"].str[:7]
    out["Mean"] = s["mean"].map(lambda v: _fmt(v))
    out["Std dev"] = s["std"].map(lambda v: _fmt(v))
    out["Min – Max"] = [f"{_fmt(a)} – {_fmt(b)}" for a, b in zip(s["min"], s["max"], strict=True)]
    out["Percentile now"] = s["lastPercentile"].map(lambda v: _fmt(v, ".0f", "th"))
    out["Z-score now"] = s["lastZScore"].map(lambda v: _fmt(v, "+.2f"))
    out["Trend / yr"] = [
        _fmt(v, "+.3f") + ("" if not ts.is_significant(p) else " *")
        for v, p in zip(s["trendPerYear"], s["trendPValue"], strict=True)
    ]
    out["ADF stat"] = s["adfStat"].map(lambda v: _fmt(v, ".2f"))
    out["Stationary (5%)"] = s["stationary"].map(lambda v: "—" if v is None else "Yes" if v else "No")
    out.index.name = "Series"
    return out


def regression_table(rep: MacroLabReport) -> pd.DataFrame | None:
    if not rep.regression:
        return None
    c = rep.regression["coefficients"].copy()
    c["Significant (5%)"] = c["p"].map(lambda p: "Yes" if ts.is_significant(p) else "No")
    c = c.rename(columns={"coef": "Coefficient", "seHAC": "Std err (HAC)", "t": "t-stat", "p": "p-value"})
    return c.round(4)


def cycle_summary_table(rep: MacroLabReport) -> pd.DataFrame:
    rows = []
    for var, c in rep.cycles.items():
        nt = c["nextTurningPoint"]
        rows.append(
            {
                "Series": var,
                "Phase": c["phase"],
                "Gap now": _fmt(c["currentGap"], "+.2f", "%" if c["logTransformed"] else ""),
                "Gap percentile": _fmt(c["gapPercentile"], ".0f", "th"),
                "Dominant cycle": _fmt(c["dominantPeriodYears"], ".1f", " y"),
                "Avg peak-to-peak": _fmt(c["avgPeakToPeakYears"], ".1f", " y"),
                "Last peak": c["peaks"][-1].strftime("%b %Y") if c["peaks"] else "—",
                "Last trough": c["troughs"][-1].strftime("%b %Y") if c["troughs"] else "—",
                "Indicative next turn": f"{nt['type']} ~{nt['date']:%b %Y}" if nt else "—",
            }
        )
    return pd.DataFrame(rows).set_index("Series") if rows else pd.DataFrame()


def macro_figures(rep: MacroLabReport, t: Theme, view: str | None = None) -> dict[str, Any]:
    colors, units = macro_colors(rep, t), macro_units(rep)
    unit = FREQ_UNIT[rep.freq]
    figs: dict[str, Any] = {
        "overview": charts.multi_series_chart(rep.levels, t, view or macro_default_view(rep), units, colors),
        "multiples": charts.small_multiples(rep.levels, t, units, colors),
    }
    if rep.corr_levels:
        figs["corr_levels"] = charts.correlation_heatmap(
            rep.corr_levels, t, "Correlation — levels", "Can be spurious for trending series"
        )
    if rep.corr_transformed:
        figs["corr_transformed"] = charts.correlation_heatmap(
            rep.corr_transformed,
            t,
            f"Correlation — {ts.TRANSFORMS[rep.transform].lower()}",
            "Stationary basis (preferred for inference)",
        )
    figs["lead_lag"] = {
        d: charts.lead_lag_chart(ll, t, d, rep.target or "", unit) for d, ll in rep.lead_lag.items()
    }
    figs["cycles"] = {v: charts.cycle_chart(rep.levels[v], c, t, v, colors[v]) for v, c in rep.cycles.items()}
    figs["spectra"] = {
        v: charts.spectrum_chart(
            c["spectrum"], t, c["dominantPeriodYears"], (CYCLE_MIN_PERIOD_YEARS, CYCLE_MAX_PERIOD_YEARS)
        )
        for v, c in rep.cycles.items()
    }
    if rep.forecast and rep.target:
        figs["forecast"] = charts.forecast_chart(
            rep.levels[rep.target], rep.forecast, t, rep.target, colors[rep.target]
        )
    if rep.regression and rep.target:
        basis = rep.transformed if rep.recommended_basis == "transformed" else rep.levels
        figs["fit"] = charts.fitted_chart(basis[rep.target], rep.regression["fitted"], t, rep.target)
    return figs


def macro_kpis(rep: MacroLabReport) -> list[Kpi]:
    k = [
        Kpi(
            "Series analysed", str(len(rep.series)), f"{len(rep.levels)} {FREQ_UNIT[rep.freq]}ly observations"
        ),
        Kpi(
            "Sample",
            f"{rep.levels.index[0]:%Y}–{rep.levels.index[-1]:%Y}",
            f"to {rep.levels.index[-1]:%b %Y}",
        ),
    ]
    if rep.regression:
        k.append(Kpi("Model fit (R²)", f"{rep.regression['r2']:.0%}", f"target: {rep.target}"))
    if rep.target in rep.cycles:
        c = rep.cycles[rep.target]
        k.append(Kpi("Target cycle phase", c["phase"], c["phaseDescription"]))
    if rep.forecast:
        k.append(Kpi("Forecast skill", f"{rep.forecast['skill']:+.0%}", "vs naive, out-of-sample"))
    return k


def macro_lab_html(rep: MacroLabReport) -> str:
    t = LIGHT
    f = macro_figures(rep, t)
    sections = [
        Section(
            "Overview",
            figures=[f["overview"], f["multiples"]],
            tables=[("Descriptive statistics", macro_stats_table(rep))],
        ),
    ]
    corr_figs = [x for x in (f.get("corr_transformed"), f.get("corr_levels")) if x is not None]
    if corr_figs:
        sections.append(
            Section("Correlation", f"Inference basis: **{rep.recommended_basis}**.", figures=corr_figs)
        )
    if f["lead_lag"]:
        sections.append(Section("Lead / lag", figures=list(f["lead_lag"].values())))
    reg = regression_table(rep)
    if reg is not None:
        sections.append(
            Section("Regression", figures=[f["fit"]], tables=[("Coefficients (Newey-West HAC)", reg)])
        )
    if f["cycles"]:
        sections.append(
            Section(
                "Cycle analysis",
                figures=list(f["cycles"].values()),
                tables=[("Cycle summary", cycle_summary_table(rep))],
            )
        )
    if "forecast" in f:
        sections.append(Section("Forecast", figures=[f["forecast"]]))
    return build_html_report(
        title="Multi-variable macro analysis",
        subtitle=" · ".join(s.label for s in rep.series),
        kpis=macro_kpis(rep),
        findings=[x.text for x in rep.findings],
        sections=sections,
        methodology=rep.methodology,
        caveats=rep.caveats,
        sources=rep.sources,
    )


# =========================================================================== Rate sensitivity


def rate_company_table(rep: RateSensitivityReport) -> pd.DataFrame:
    c = rep.companies
    out = pd.DataFrame(
        {
            "Company": c["company"],
            "FY": c["fiscalYear"],
            "Ccy": c["currency"],
            "Effective rate": c["effectiveRate"].map(lambda v: _fmt(v, ".2f", "%")),
            "Spread vs bench": c["spread"].map(lambda v: _fmt(v, "+.2f", " pp")),
            "Interest / revenue": c["interestBurden"].map(lambda v: _fmt(v, ".2f", "%")),
            "EBIT coverage": c["coverage"].map(lambda v: _fmt(v, ".1f", "×")),
            "Debt / EBITDA": c["debtToEbitda"].map(lambda v: _fmt(v, ".1f", "×")),
            "Total debt (bn)": (c["totalDebt"] / 1e9).map(lambda v: _fmt(v, ",.1f")),
            "Status": np.where(
                c["excluded"], "Excluded (data quality)", np.where(c["flagged"], "Flagged year", "OK")
            ),
        },
        index=c.index,
    )
    out.index.name = "Ticker"
    return out


def rate_industry_table(rep: RateSensitivityReport) -> pd.DataFrame:
    ind = rep.industry.copy()
    cols = {
        "medianRate": "Median rate %",
        "q1Rate": "Q1 %",
        "q3Rate": "Q3 %",
        "benchmark": "Benchmark %",
        "spread": "Spread pp",
        "medianCoverage": "Median coverage ×",
        "medianBurden": "Interest/revenue %",
        "companies": "Companies",
    }
    if "aggregateRate" in ind:
        cols["aggregateRate"] = "Debt-weighted rate %"
    ind = ind[list(cols)].rename(columns=cols).round(2)
    ind.index.name = "Fiscal year"
    return ind


def passthrough_table(rep: RateSensitivityReport) -> pd.DataFrame | None:
    rows = []
    for name, res in (("Same-year", rep.passthrough), ("With 1-year lag", rep.passthrough_lagged)):
        if res is None:
            continue
        for var, r in res["coefficients"].iterrows():
            if var == "const":
                continue
            rows.append(
                {
                    "Model": name,
                    "Term": {"dBench": "Δ benchmark (t)", "dBenchLag1": "Δ benchmark (t−1)"}[str(var)],
                    "β": round(r["coef"], 3),
                    "Std err (clustered)": round(r["seCluster"], 3),
                    "p-value": round(r["p"], 3),
                    "Significant": "Yes" if ts.is_significant(r["p"]) else "No",
                    "R²": round(res["r2"], 3),
                    "n / companies": f"{res['n']} / {res['groups']}",
                }
            )
    return pd.DataFrame(rows).set_index(["Model", "Term"]) if rows else None


def rate_figures(rep: RateSensitivityReport, t: Theme) -> dict[str, go.Figure]:
    figs = {
        "rates": charts.rate_vs_benchmark_chart(
            rep.panel[~rep.panel["symbol"].isin(rep.companies.index[rep.companies["excluded"]])],
            rep.industry,
            rep.benchmark.label,
            t,
        ),
        "spread": charts.spread_chart(rep.industry, rep.benchmark.label, t),
        "coverage": charts.coverage_chart(rep.companies, t),
    }
    if not rep.scenarios.empty:
        figs["scenarios"] = charts.scenario_heatmap(rep.scenarios, t, rep.scenario_beta)
    return figs


def rate_kpis(rep: RateSensitivityReport) -> list[Kpi]:
    ind = rep.industry.dropna(subset=["medianRate"])
    k: list[Kpi] = []
    if not ind.empty:
        last = ind.iloc[-1]
        k += [
            Kpi("Median cost of debt", f"{last['medianRate']:.2f}%", f"FY{ind.index[-1]}"),
            Kpi(rep.benchmark.label, f"{last['benchmark']:.2f}%", "fiscal-year average"),
            Kpi("Spread", f"{last['spread']:+.2f} pp", "median − benchmark"),
            Kpi("Median coverage", _fmt(last["medianCoverage"], ".1f", "×"), "EBIT ÷ interest"),
        ]
    if rep.passthrough is not None:
        c = rep.passthrough["coefficients"].loc["dBench"]
        k.append(Kpi("Pass-through β", f"{c['coef']:+.2f}", f"p = {c['p']:.2f}"))
    return k


def rate_sensitivity_html(rep: RateSensitivityReport) -> str:
    f = rate_figures(rep, LIGHT)
    sections = [
        Section(
            "Borrowing costs vs benchmark",
            figures=[f["rates"], f["spread"]],
            tables=[("Industry by fiscal year", rate_industry_table(rep))],
        ),
        Section(
            "Company detail",
            figures=[f["coverage"]],
            tables=[("Latest fiscal year", rate_company_table(rep))],
        ),
    ]
    pt = passthrough_table(rep)
    scen_figs = [f["scenarios"]] if "scenarios" in f else []
    sections.append(
        Section(
            "Pass-through & scenarios",
            figures=scen_figs,
            tables=[("Pass-through regression", pt)] if pt is not None else [],
        )
    )
    return build_html_report(
        title=f"Rate sensitivity — {rep.universe_description}",
        subtitle=f"Benchmark: {rep.benchmark.label} ({rep.benchmark.source})",
        kpis=rate_kpis(rep),
        findings=rep.findings,
        sections=sections,
        methodology=rep.methodology,
        caveats=rep.caveats,
        sources=[rep.benchmark.source, "FMP income statements & balance sheets (annual)"],
    )


# =========================================================================== Relative valuation


def benchmark_table(rep: RelativeValuationReport) -> pd.DataFrame:
    rows = [
        {
            "Benchmark": b.name,
            "P/E": _fmt(b.pe, ".1f", "×"),
            f"{rep.symbol} premium": _fmt(b.premium, "+.0%"),
            "Companies": str(b.n) if b.n is not None else "—",
            "Basis": b.basis,
            "Primary": "●" if b.name == rep.primary_benchmark else "",
        }
        for b in rep.benchmarks
    ]
    return pd.DataFrame(rows).set_index("Benchmark")


def peer_table(rep: RelativeValuationReport) -> pd.DataFrame:
    p = rep.peers
    if p.empty:
        return pd.DataFrame()
    out = pd.DataFrame(
        {
            "Company": p["company"],
            "Set": p["set"],
            "Mkt cap ($bn)": (p["marketCap"] / 1e9).map(lambda v: _fmt(v, ",.1f")),
            "P/E (TTM)": [
                _fmt(v, ".1f", "×") if m else ("NM" if v is not None else "—")
                for v, m in zip(p["pe"], p["peMeaningful"], strict=True)
            ],
            "EV/EBITDA": p["evToEbitda"].map(lambda v: _fmt(v, ".1f", "×")),
            "EPS growth": p["epsGrowth"].map(lambda v: _fmt(v, "+.0%")),
            "ROE": p["roe"].map(lambda v: _fmt(v, ".0%")),
            "Net margin": p["netMargin"].map(lambda v: _fmt(v, ".1%")),
            "Origin": p["origin"],
        }
    )
    out.index = p["symbol"]
    out.index.name = "Ticker"
    return out


def relval_figures(
    rep: RelativeValuationReport, t: Theme, show_industry: bool = False
) -> dict[str, go.Figure]:
    figs = {"benchmarks": charts.pe_benchmark_chart(rep.stock["pe"], rep.symbol, rep.benchmarks, t)}
    if not rep.peers.empty:
        figs["scatter_growth"] = charts.peer_scatter(rep.peers, rep.stock, t, "epsGrowth", "EPS growth")
        figs["scatter_roe"] = charts.peer_scatter(rep.peers, rep.stock, t, "roe", "Return on equity")
    if not rep.history.empty:
        figs["history"] = charts.pe_history_chart(
            rep.history, rep.history_stats, t, rep.symbol, show_industry
        )
        figs["relative"] = charts.relative_pe_chart(rep.history, t, rep.symbol)
    return figs


def relval_kpis(rep: RelativeValuationReport) -> list[Kpi]:
    pe = rep.stock.get("pe")
    k = [Kpi(f"{rep.symbol} P/E (TTM)", _fmt(pe, ".1f", "×"), f"price ${_fmt(rep.price, ',.2f')}")]
    for b in rep.benchmarks:
        if b.name in (rep.primary_benchmark, "US large-cap market"):
            k.append(
                Kpi(
                    b.name.replace(" — median", " median"),
                    _fmt(b.pe, ".1f", "×"),
                    f"premium {_fmt(b.premium, '+.0%')}",
                )
            )
    h = rep.history_stats
    if h.get("z") is not None:
        k.append(
            Kpi(
                "vs own history",
                f"{h['z']:+.1f}σ",
                f"{h['percentile']:.0f}th percentile since {h['start'][:4]}",
            )
        )
    k.append(Kpi("Relative valuation", rep.verdict, rep.verdict_detail))
    return k


def relative_valuation_html(rep: RelativeValuationReport) -> str:
    f = relval_figures(rep, LIGHT, show_industry=True)
    sections = [
        Section(
            "Multiple vs benchmarks", figures=[f["benchmarks"]], tables=[("Benchmarks", benchmark_table(rep))]
        )
    ]
    if "scatter_growth" in f:
        sections.append(
            Section(
                "Peer comparison",
                figures=[f["scatter_growth"], f["scatter_roe"]],
                tables=[("Peer group", peer_table(rep))],
            )
        )
    if "history" in f:
        sections.append(Section("Valuation through time", figures=[f["history"], f["relative"]]))
    return build_html_report(
        title=f"{rep.company} ({rep.symbol}) — relative valuation",
        subtitle=f"{rep.sector} · {rep.industry} · {rep.exchange}. Verdict: **{rep.verdict}** — {rep.verdict_detail}. "
        "A relative-valuation reading, not an investment rating.",
        kpis=relval_kpis(rep),
        findings=rep.findings,
        sections=sections,
        methodology=rep.methodology,
        caveats=rep.caveats,
        sources=[
            "FMP ratios-ttm, key-metrics-ttm, statements, profile, screener",
            "FMP sector/industry P/E snapshots and history",
            "FMP daily price history",
        ],
    )


# =========================================================================== Fundamentals panel


def panel_kpis(rep: Any) -> list[Kpi]:
    c = rep.coverage
    return [
        Kpi("Companies", f"{c['companies']:,}", f"{c['sectors']} sectors"),
        Kpi("Company-years", f"{c['companyYears']:,}", f"fiscal years {c['yearRange']}"),
        Kpi("Complete years", str(c["completeYears"]), "≥ 80% of universe reported"),
        Kpi("Reporting currencies", str(len(c["currencies"])), ", ".join(c["currencies"][:4])),
    ]


def leaderboard_table(lb: pd.DataFrame, top: int = 50) -> pd.DataFrame:
    if lb.empty:
        return lb
    d = lb.head(top)
    out = pd.DataFrame(
        {
            "Company": d["company"],
            "Sector": d["sector"],
            "Revenue CAGR": d["revenueCAGR"].map(lambda v: _fmt(v, "+.1%")),
            "EPS CAGR": d["epsCAGR"].map(lambda v: _fmt(v, "+.1%")),
            "Avg op. margin": d["avgOperatingMargin"].map(lambda v: _fmt(v, ".1%")),
            "Op. margin trend": d["operatingMarginTrendPpYr"].map(lambda v: _fmt(v, "+.2f", " pp/yr")),
            "Margin volatility": d["operatingMarginVolatility"].map(lambda v: _fmt(v, ".1%")),
            "Sector pctile (growth)": d["revenueCAGRSectorPctile"].map(lambda v: _fmt(v, ".0f", "th")),
            "Ccy": d["currency"],
            "Note": d["smallBase"].map(lambda v: "small base" if v else ""),
        }
    )
    out.index = d["symbol"]
    out.index.name = "Ticker"
    return out


def panel_html(rep: Any, metric: str, highlight: list[str]) -> str:
    from src.analysis.fundamentals_panel import METRICS, growth_leaderboard, sector_trend

    t = LIGHT
    yrs = rep.complete_years
    meta = METRICS[metric]
    med, cnt = sector_trend(rep.panel, metric, yrs)
    figs = [
        charts.sector_heatmap(med, cnt, t, meta["label"], meta["fmt"], meta["kind"] == "change"),
        charts.sector_lines(med, t, meta["label"], meta["fmt"], highlight),
    ]
    sections = [Section(f"Sector trends — {meta['label']}", figures=figs)]
    if len(yrs) >= 2:
        lb = growth_leaderboard(rep.panel, yrs[0], yrs[-1])
        sections.append(
            Section(
                f"Growth leaderboard {yrs[0]}–{yrs[-1]}",
                figures=[charts.leaderboard_scatter(lb, t, None)],
                tables=[("Top 50 by revenue CAGR", leaderboard_table(lb))],
            )
        )
    return build_html_report(
        title="Fundamentals panel — 10-year income statement analysis",
        subtitle=f"{rep.coverage['companies']} companies · fiscal years {rep.coverage['yearRange']}",
        kpis=panel_kpis(rep),
        findings=rep.findings,
        sections=sections,
        methodology=rep.methodology,
        caveats=rep.caveats,
        sources=[
            "FMP annual income statements (local Parquet store, DuckDB)",
            "FMP company screener (universe snapshot)",
        ],
    )


# =========================================================================== Stock analysis (time series)


def _f(v: Any) -> float | None:
    """Finite float or None (DataFrame cells come back as NaN, not None)."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


_STATEMENT_ROWS = [
    ("Revenue", "inc_revenue", "money"),
    ("Gross profit", "inc_grossProfit", "money"),
    ("Operating income", "inc_operatingIncome", "money"),
    ("EBITDA", "inc_ebitda", "money"),
    ("Net income", "inc_netIncome", "money"),
    ("Diluted EPS", "inc_epsDiluted", "price"),
    ("Operating cash flow", "cf_operatingCashFlow", "money"),
    ("Capital expenditure", "x_capex", "money"),
    ("Free cash flow", "cf_freeCashFlow", "money"),
    ("Dividends paid", "x_dividends", "money"),
    ("Total assets", "bs_totalAssets", "money"),
    ("Cash & short-term investments", "bs_cashAndShortTermInvestments", "money"),
    ("Total debt", "bs_totalDebt", "money"),
    ("Net debt", "bs_netDebt", "money"),
    ("Shareholders' equity", "bs_totalStockholdersEquity", "money"),
    ("Market cap (FY end)", "km_marketCap", "money_quote"),
]


def _years(df: pd.DataFrame, last_n: int | None) -> list[int]:
    yrs = [int(y) for y in df.index]
    return yrs[-last_n:] if last_n else yrs


def statement_table(c: Any, last_n: int | None = None) -> pd.DataFrame:
    """Key statement lines by fiscal year, money in the reporting currency."""
    from src.reporting.formatting import money, price

    yrs = _years(c.raw, last_n)
    rows = {}
    for label, col, kind in _STATEMENT_ROWS:
        if col not in c.raw:
            continue
        s = pd.to_numeric(c.raw[col], errors="coerce")
        ccy = c.quote_currency if kind == "money_quote" else c.reporting_currency
        f = price if kind == "price" else money
        rows[label] = {f"FY{y}": f(s.get(y), ccy) for y in yrs}
    return pd.DataFrame.from_dict(rows, orient="index")


def indicator_history_table(c: Any, keys: list[str], last_n: int | None = None) -> pd.DataFrame:
    from src.analysis.indicators import GROUPS, INDICATORS
    from src.reporting.formatting import value

    yrs = _years(c.values, last_n)
    rows = []
    for k in keys:
        ind = INDICATORS[k]
        if ind.market:
            continue
        s = c.values[k] if k in c.values else pd.Series(dtype=float)
        rows.append(
            {
                "Indicator": ind.label,
                "Group": GROUPS[ind.group],
                **{f"FY{y}": value(s.get(y), ind.fmt) for y in yrs},
            }
        )
    return pd.DataFrame(rows).set_index("Indicator") if rows else pd.DataFrame()


def market_table(c: Any) -> pd.DataFrame:
    from src.analysis.indicators import INDICATORS
    from src.reporting.formatting import value

    m = c.market
    rows = []
    for k in ("ret1y", "cagr3y", "cagr5y", "volatility", "maxDrawdown", "beta"):
        ind = INDICATORS[k]
        note = ind.description
        if k == "beta" and m.get("beta") is not None:
            note = f"p = {m['betaP']:.3f}, n = {m['betaN']} weeks, HAC s.e. {m['betaSE']:.2f}"
        rows.append(
            {"Metric": ind.label, "Group": ind.group.title(), "Value": value(m.get(k), ind.fmt), "Note": note}
        )
    return pd.DataFrame(rows).set_index("Metric")


def stock_kpis(c: Any) -> list[Kpi]:
    from src.reporting.formatting import money, price, value

    last = c.values.iloc[-1] if not c.values.empty else pd.Series(dtype=float)
    fy = c.latest_year
    rev = pd.to_numeric(c.raw.get("inc_revenue"), errors="coerce") if "inc_revenue" in c.raw else None
    m = c.market
    pe = c.ttm.get("pe")
    return [
        Kpi("Price", price(c.price, c.quote_currency), f"1-year return {value(m.get('ret1y'), 'pct')}"),
        Kpi("Market cap", money(c.market_cap, c.quote_currency), c.exchange),
        Kpi(
            "P/E (TTM)",
            "n/a (currency mismatch)" if c.currency_mismatch else value(pe, "x"),
            f"EV/EBITDA {value(c.ttm.get('evToEbitda'), 'x')}",
        ),
        Kpi(
            f"Revenue FY{fy}",
            money(rev.get(fy) if rev is not None else None, c.reporting_currency),
            f"growth {value(last.get('revenueGrowth'), 'pct')}",
        ),
        Kpi(
            "Return on equity",
            value(last.get("roe"), "pct"),
            f"net margin {value(last.get('netMargin'), 'pct')}",
        ),
        Kpi(
            "Gearing (net debt ÷ equity)",
            value(last.get("gearing"), "pct"),
            f"net debt ÷ EBITDA {value(last.get('netDebtToEbitda'), 'x')}",
        ),
        Kpi(
            "Volatility / beta",
            f"{value(m.get('volatility'), 'pct')} / {value(m.get('beta'), 'ratio')}",
            f"max drawdown {value(m.get('maxDrawdown'), 'pct')}",
        ),
    ]


def _group_panels(
    keys: list[str], data_for: Any, median_for: Any = None, agg_for: Any = None
) -> dict[str, list]:
    from src.analysis.indicators import INDICATORS

    out: dict[str, list[dict[str, Any]]] = {"return": [], "risk": [], "valuation": []}
    for k in keys:
        ind = INDICATORS[k]
        if ind.market:
            continue
        out[ind.group].append(
            {
                "label": ind.label,
                "fmt": ind.fmt,
                "data": data_for(k),
                "median": median_for(k) if median_for else None,
                "aggregate": agg_for(k) if agg_for else None,
            }
        )
    return out


def stock_figures(c: Any, keys: list[str], t: Theme, last_n: int | None = None) -> dict[str, go.Figure]:
    yrs = _years(c.values, last_n)
    colors = {c.symbol: t.series[0]}

    def data_for(k: str) -> pd.DataFrame:
        return c.values[c.values.index.isin(yrs)][[k]].rename(columns={k: c.symbol})

    figs: dict[str, go.Figure] = {}
    for group, panels in _group_panels(keys, data_for).items():
        if panels:
            figs[group] = charts.indicator_panels(
                panels,
                t,
                c.symbol,
                colors,
                f"{c.symbol} — {group} indicators",
                f"Fiscal years FY{yrs[0]}–FY{yrs[-1]} · each panel on its own scale",
            )
    raw = c.raw[c.raw.index.isin(yrs)]
    figs["statements"] = charts.statement_bars(raw, t, c.symbol, c.reporting_currency)
    if not c.prices.empty:
        figs["price"] = charts.price_ma_chart(c.prices, t, c.symbol, c.quote_currency)
        figs["drawdown"] = charts.drawdown_chart(c.prices.to_frame(c.symbol), t, colors, None)
    return figs


def stock_analysis_html(c: Any, keys: list[str], findings: list[str], last_n: int | None = None) -> str:
    f = stock_figures(c, keys, LIGHT, last_n)
    sections = [
        Section(
            "Financial statements",
            figures=[f["statements"]],
            tables=[(f"Key lines ({c.reporting_currency})", statement_table(c, last_n))],
        )
    ]
    for g, title in (
        ("return", "Return indicators"),
        ("risk", "Risk indicators"),
        ("valuation", "Valuation"),
    ):
        if g in f:
            sections.append(Section(title, figures=[f[g]]))
    sections.append(
        Section("Indicator history", tables=[("Annual indicators", indicator_history_table(c, keys, last_n))])
    )
    if "price" in f:
        sections.append(
            Section(
                "Price return & risk",
                figures=[f["price"], f["drawdown"]],
                tables=[("Market", market_table(c))],
            )
        )
    return build_html_report(
        title=f"{c.company} ({c.symbol}) — fundamentals over time",
        subtitle=f"{c.sector} · {c.industry} · {c.exchange}. Quote {c.quote_currency}, reporting {c.reporting_currency}.",
        kpis=stock_kpis(c),
        findings=findings,
        sections=sections,
        methodology=[
            "Indicators are computed per fiscal year from FMP annual income statement, balance sheet, cash flow, "
            "ratios and key metrics (Starter tier: up to 10 years; quarterly ratios are Premium).",
            "Balance-sheet ratios use period-end values; growth is blank when the prior-year base is <= 0.",
            "Valuation multiples in the time series use the fiscal-year-end market cap; KPI tiles use TTM.",
            "Price metrics use daily closes (dividends excluded); beta = weekly-return OLS slope vs SPY, HAC s.e.",
        ],
        caveats=stock_caveats(c),
        sources=["FMP statements, ratios, key metrics (annual + TTM), profile, daily prices"],
    )


def stock_caveats(c: Any) -> list[str]:
    out = list(c.notes)
    if c.currency_mismatch:
        out.insert(
            0,
            f"Currency mismatch: quotes in {c.quote_currency}, reports in {c.reporting_currency}. Valuation "
            "multiples are unavailable (no FX conversion); statement money is shown in the reporting currency.",
        )
    return out


# =========================================================================== Industry comparison


def _agg_rows(rep: Any, keys: list[str], basis: str) -> pd.DataFrame:
    from src.analysis.industry_analysis import peer_aggregates

    return peer_aggregates(rep, keys, basis)


def comparison_table(rep: Any, keys: list[str], basis: str = "ttm") -> pd.DataFrame:
    """Companies (subject first) + peer median / mean / aggregate / subject percentile rows."""
    from src.analysis.indicators import INDICATORS
    from src.analysis.industry_analysis import cap_bucket, snapshot
    from src.reporting.formatting import money, value

    snap = snapshot(rep, keys, basis)
    agg = _agg_rows(rep, keys, basis)
    labels = {k: INDICATORS[k].label for k in keys}
    rows: dict[str, dict[str, str]] = {}
    for sym, c in rep.companies.items():
        row = {
            "Company": c.company,
            "Mkt cap": money(c.market_cap, c.quote_currency),
            "Size": cap_bucket(c.market_cap),
            "FY": f"FY{c.latest_year}" if c.latest_year else "—",
        }
        for k in keys:
            row[labels[k]] = value(snap.at[sym, k], INDICATORS[k].fmt)
        rows[sym + (" ★" if sym == rep.symbol else "")] = row
    n_peers = len(rep.peers)
    for stat, name in (
        ("median", "Peer median"),
        ("mean", "Peer mean"),
        ("aggregate", "Peer aggregate (Σ÷Σ)"),
    ):
        row = {"Company": f"{name} · {n_peers} peers ex-{rep.symbol}", "Mkt cap": "", "Size": "", "FY": ""}
        for k in keys:
            row[labels[k]] = value(_f(agg.at[k, stat]), INDICATORS[k].fmt)
        rows[name] = row
    prow = {"Company": f"{rep.symbol} beats % of peers", "Mkt cap": "", "Size": "", "FY": ""}
    for k in keys:
        p = _f(agg.at[k, "percentile"])
        prow[labels[k]] = f"{p:.0%}" if p is not None else "—"
    rows["Percentile"] = prow
    return pd.DataFrame.from_dict(rows, orient="index")


def industry_panels(rep: Any, keys: list[str], last_n: int | None = None) -> dict[str, list[dict[str, Any]]]:
    from src.analysis.industry_analysis import aggregate_history, indicator_history

    cache: dict[str, pd.DataFrame] = {}

    def hist(k: str) -> pd.DataFrame:
        h = indicator_history(rep, k)
        yrs = _years(h, last_n) if not h.empty else []
        return h[h.index.isin(yrs)]

    def agg(k: str) -> pd.DataFrame:
        if k not in cache:
            a = aggregate_history(rep, k)
            yrs = _years(a, last_n) if not a.empty else []
            cache[k] = a[a.index.isin(yrs)]
        return cache[k]

    return _group_panels(keys, hist, lambda k: agg(k)["median"], lambda k: agg(k)["aggregate"])


def industry_figures(
    rep: Any, keys: list[str], t: Theme, highlight: list[str], basis: str = "ttm", last_n: int | None = None
) -> dict[str, Any]:
    from src.analysis.indicators import INDICATORS
    from src.analysis.industry_analysis import price_panel, price_table, snapshot

    colors = charts.entity_colors(t, rep.symbol, highlight)
    figs: dict[str, Any] = {"colors": colors}
    for group, panels in industry_panels(rep, keys, last_n).items():
        if panels:
            figs[f"ts_{group}"] = charts.indicator_panels(
                panels,
                t,
                rep.symbol,
                colors,
                f"{group.title()} indicators over time — {rep.symbol} vs peers",
                "Grey = other peers · dotted = peer median · dashed = peer aggregate (Σ÷Σ) · peers exclude subject",
            )
    snap = snapshot(rep, keys, basis)
    agg = _agg_rows(rep, keys, basis)
    for k in keys:
        ind = INDICATORS[k]
        figs[f"rank_{k}"] = charts.peer_rank_bar(
            snap[k],
            t,
            rep.symbol,
            colors,
            ind.label,
            ind.fmt,
            _f(agg.at[k, "median"]),
            _f(agg.at[k, "aggregate"]),
            ind.higher_is_better,
        )
    prices = price_panel(rep)
    if not prices.empty:
        bench = f"{rep.bench_symbol} (market)"
        figs["cumret"] = charts.cumulative_return_chart(prices, t, rep.symbol, colors, bench)
        figs["rollvol"] = charts.rolling_vol_chart(prices, t, colors, bench)
        figs["drawdown"] = charts.drawdown_chart(prices, t, colors, bench)
        figs["scatter"] = charts.return_risk_scatter(
            price_table(rep), t, rep.symbol, colors, "cagr3y", "3-year price CAGR"
        )
    return figs


def peers_display_table(rep: Any) -> pd.DataFrame:
    from src.analysis.industry_analysis import peers_table
    from src.reporting.formatting import money

    p = peers_table(rep)
    return pd.DataFrame(
        {
            "Company": p["company"],
            "Industry": p["industry"],
            "Country": p["country"],
            "Market cap": [money(v) for v in p["marketCap"]],
            "Size": p["capBucket"],
            "Reports in": p["reportingCurrency"],
            "FX mismatch": p["currencyMismatch"].map(lambda v: "⚠️ yes" if v else ""),
            "Latest FY": p["latestFY"],
            "Source": p["origin"],
        },
        index=p.index,
    )


def industry_kpis(rep: Any, basis: str = "ttm") -> list[Kpi]:
    from src.reporting.formatting import money, value

    agg = _agg_rows(rep, ["pe", "evToEbitda", "roe", "gearing", "volatility"], basis)
    caps = [c.market_cap for c in rep.companies.values() if c.market_cap]

    def a(k: str, stat: str) -> float | None:
        return _f(agg.at[k, stat])

    return [
        Kpi("Peer group", f"{len(rep.peers)} peers", f"{rep.profile['label']} profile"),
        Kpi(
            "Combined market cap", money(sum(caps)), f"median {money(float(np.median(caps)))}" if caps else ""
        ),
        Kpi(
            f"{rep.symbol} P/E vs industry",
            f"{value(a('pe', 'subject'), 'x')} vs {value(a('pe', 'aggregate'), 'x')}",
            f"peer aggregate Σcap÷Σearnings · median {value(a('pe', 'median'), 'x')}",
        ),
        Kpi(
            "EV/EBITDA vs peer median",
            f"{value(a('evToEbitda', 'subject'), 'x')} vs {value(a('evToEbitda', 'median'), 'x')}",
            "TTM" if basis == "ttm" else "latest FY",
        ),
        Kpi(
            "ROE vs peer median",
            f"{value(a('roe', 'subject'), 'pct')} vs {value(a('roe', 'median'), 'pct')}",
            "latest fiscal year",
        ),
        Kpi(
            "Gearing vs peer median",
            f"{value(a('gearing', 'subject'), 'pct')} vs {value(a('gearing', 'median'), 'pct')}",
            "net debt ÷ equity",
        ),
        Kpi(
            "Volatility vs peer median",
            f"{value(a('volatility', 'subject'), 'pct')} vs {value(a('volatility', 'median'), 'pct')}",
            "annualised, 3 years",
        ),
    ]


def industry_comparison_html(
    rep: Any,
    keys: list[str],
    highlight: list[str],
    findings: list[str],
    basis: str = "ttm",
    last_n: int | None = None,
) -> str:
    from src.analysis.indicators import INDICATORS

    f = industry_figures(rep, keys, LIGHT, highlight, basis, last_n)
    sections = [Section("Peer universe", tables=[("Companies", peers_display_table(rep))])]
    for g in ("return", "risk", "valuation"):
        gk = [k for k in keys if INDICATORS[k].group == g]
        if not gk:
            continue
        figs = [f[f"ts_{g}"]] if f"ts_{g}" in f else []
        sections.append(
            Section(
                f"{g.title()} comparison",
                figures=figs,
                tables=[(f"{g.title()} — latest", comparison_table(rep, gk, basis))],
            )
        )
    if "cumret" in f:
        sections.append(Section("Market return & risk", figures=[f["cumret"], f["scatter"], f["rollvol"]]))
    return build_html_report(
        title=f"{rep.subject.company} ({rep.symbol}) — industry comparison",
        subtitle=f"{rep.subject.sector} · {rep.subject.industry} · {rep.profile['label']} indicator profile.",
        kpis=industry_kpis(rep, basis),
        findings=findings,
        sections=sections,
        methodology=rep.methodology,
        caveats=rep.caveats,
        sources=[
            "FMP company screener / stock-peers (universe)",
            "FMP statements, ratios, key metrics (annual + TTM), profile",
            "FMP daily price history (incl. SPY benchmark)",
        ],
    )
