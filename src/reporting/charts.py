# src/reporting/charts.py
#
# Plotly figure factories for every analysis. Pure presentation: each takes
# analysis output + a Theme and returns a go.Figure, so the Streamlit pages
# and the HTML report exporter render identical charts.
#
# Conventions (dataviz skill): one y-axis per plot — measures with different
# units go to small multiples or a common base, never a dual axis; thin 2px
# lines; highlight-one/grey-the-rest for "where does X sit" questions;
# diverging scales are blue↔red through a neutral grey; hover on by default.

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.reporting.theme import Theme

UNIT_SUFFIX = {"pct": "%", "ratio": "×"}


def _layout(
    fig: go.Figure, t: Theme, title: str, subtitle: str | None = None, height: int = 420
) -> go.Figure:
    text = f"<b>{title}</b>"
    if subtitle:
        text += f"<br><span style='font-size:12px;color:{t.ink_secondary}'>{subtitle}</span>"
    fig.update_layout(
        template=f"research_{t.mode}",
        title={"text": text},
        height=height,
        barcornerradius=4,
        margin={"t": 92 if subtitle else 72},
        legend={"y": 1.02, "yanchor": "bottom"},
    )
    return fig


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def series_color(t: Theme, i: int) -> str:
    return t.series[i % len(t.series)]


# =========================================================================== Macro Lab


def multi_series_chart(
    levels: pd.DataFrame, t: Theme, view: str, units: dict[str, str], colors: dict[str, str]
) -> go.Figure:
    """All series on ONE axis. view: 'levels' (only when units match),
    'zscore' (standardised), or 'index' (=100 at first observation)."""
    if view == "zscore":
        data = (levels - levels.mean()) / levels.std(ddof=1)
        y_title, sub = (
            "Standard deviations from mean",
            "Standardised (z-score) so different units share one axis",
        )
    elif view == "index":
        data = levels.divide(levels.iloc[0].replace(0, np.nan)) * 100
        y_title, sub = f"Index ({levels.index[0]:%b %Y} = 100)", "Rebased to a common starting value"
    else:
        data = levels
        unit = next(iter(set(units.values())), "")
        y_title, sub = {"pct": "Percent", "ratio": "Multiple (×)"}.get(unit, "Level"), "Levels (shared unit)"
    fig = go.Figure()
    for col in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[col],
                name=col,
                mode="lines",
                line={"color": colors[col]},
                hovertemplate="%{y:,.2f}<extra>" + col + "</extra>",
            )
        )
    if view == "zscore":
        fig.add_hline(y=0, line={"color": t.baseline, "width": 1})
    fig.update_yaxes(title_text=y_title)
    return _layout(fig, t, "Series overview", sub, height=440)


def small_multiples(
    levels: pd.DataFrame, t: Theme, units: dict[str, str], colors: dict[str, str]
) -> go.Figure:
    """One panel per series, own y-scale, shared time axis."""
    n = levels.shape[1]
    cols = 2 if n > 1 else 1
    rows = int(np.ceil(n / cols))
    fig = make_subplots(
        rows=rows,
        cols=cols,
        shared_xaxes=True,
        subplot_titles=list(levels.columns),
        vertical_spacing=0.14 if rows > 1 else 0.1,
        horizontal_spacing=0.08,
    )
    for i, col in enumerate(levels.columns):
        r, c = i // cols + 1, i % cols + 1
        suffix = UNIT_SUFFIX.get(units.get(col, ""), "")
        fig.add_trace(
            go.Scatter(
                x=levels.index,
                y=levels[col],
                name=col,
                mode="lines",
                line={"color": colors[col]},
                showlegend=False,
                hovertemplate="%{y:,.2f}" + suffix + "<extra>" + col + "</extra>",
            ),
            row=r,
            col=c,
        )
    for ann in fig.layout.annotations:
        ann.font = {"size": 12, "color": t.ink_secondary}
        ann.x = 0 if ann.x < 0.5 else 0.54
        ann.xanchor = "left"
    _layout(
        fig,
        t,
        "Each series on its own scale",
        "Small multiples — no dual axes",
        height=max(300, 230 * rows + 80),
    )
    fig.update_layout(hovermode="x")
    fig.update_xaxes(showgrid=False, showline=True, linecolor=t.baseline)
    fig.update_yaxes(gridcolor=t.grid)
    return fig


def correlation_heatmap(corr: dict[str, pd.DataFrame], t: Theme, title: str, subtitle: str) -> go.Figure:
    r, p, n = corr["r"], corr["p"], corr["n"]
    labels = list(r.columns)
    r_np, p_np, n_np = (m.to_numpy(dtype=float) for m in (r, p, n))
    z = r.to_numpy(dtype=float).copy()
    text = []
    hover = []
    for i, a in enumerate(labels):
        row_t, row_h = [], []
        for j, b in enumerate(labels):
            if i == j:
                z[i, j] = np.nan
                row_t.append("")
                row_h.append(a)
                continue
            rv, pv = r_np[i, j], p_np[i, j]
            stars = (
                ""
                if pd.isna(pv)
                else "***"
                if pv < 0.001
                else "**"
                if pv < 0.01
                else "*"
                if pv < 0.05
                else ""
            )
            row_t.append("" if pd.isna(rv) else f"{rv:+.2f}{stars}")
            row_h.append(
                f"<b>{rv:+.2f}</b>  r<br>{a} × {b}<br>p = {pv:.3g} · n = {int(n_np[i, j])}"
                if pd.notna(rv)
                else "n/a"
            )
        text.append(row_t)
        hover.append(row_h)
    neg, mid, pos = t.diverging
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            zmin=-1,
            zmax=1,
            zmid=0,
            colorscale=[[0, neg], [0.5, mid], [1, pos]],
            text=text,
            texttemplate="%{text}",
            textfont={"size": 12, "color": t.ink},
            hovertext=hover,
            hovertemplate="%{hovertext}<extra></extra>",
            xgap=2,
            ygap=2,
            colorbar={
                "title": {"text": "r", "font": {"color": t.ink_secondary}},
                "thickness": 10,
                "tickfont": {"color": t.ink_muted},
                "outlinewidth": 0,
            },
        )
    )
    fig.update_xaxes(showgrid=False, showline=False, tickangle=-20)
    fig.update_yaxes(showgrid=False, autorange="reversed")
    _layout(
        fig,
        t,
        title,
        subtitle + " · * p<0.05  ** p<0.01  *** p<0.001",
        height=max(360, 90 * len(labels) + 140),
    )
    fig.update_layout(hovermode="closest")
    return fig


def rolling_corr_chart(rc: pd.Series, t: Theme, a: str, b: str, window: int, unit: str) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=rc.index,
            y=rc,
            mode="lines",
            name="Rolling r",
            line={"color": t.series[0]},
            hovertemplate="r = %{y:+.2f}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line={"color": t.baseline, "width": 1})
    fig.update_yaxes(range=[-1.05, 1.05], title_text="Correlation (r)")
    return _layout(
        fig, t, f"Rolling {window}-{unit} correlation", f"{a} × {b} — is the relationship stable?", height=340
    )


def lead_lag_chart(ll: pd.DataFrame, t: Theme, driver: str, target: str, unit: str) -> go.Figure:
    if ll.empty:
        return _layout(go.Figure(), t, "Lead / lag", "Not enough overlapping data")
    best = int(ll["lag"].to_numpy()[int(np.argmax(ll["r"].abs().to_numpy()))])
    colors = [t.series[0] if lag == best else t.neutral_mark for lag in ll["lag"]]
    crit = 1.96 / np.sqrt(ll["n"].mean())
    fig = go.Figure()
    fig.add_hrect(y0=-crit, y1=crit, fillcolor=_rgba(t.ink_muted, 0.10), line_width=0)
    fig.add_trace(
        go.Bar(
            x=ll["lag"],
            y=ll["r"],
            marker={"color": colors},
            customdata=np.stack([ll["p"], ll["n"]], axis=1),
            hovertemplate="<b>r = %{y:+.2f}</b><br>lag %{x} "
            + unit
            + "s<br>p = %{customdata[0]:.3g} · n = %{customdata[1]}<extra></extra>",
            name="Cross-correlation",
        )
    )
    fig.add_hline(y=0, line={"color": t.baseline, "width": 1})
    fig.add_annotation(
        x=best,
        y=float(ll.loc[ll["lag"] == best, "r"].iloc[0]),
        text=f"peak: {best:+d}",
        showarrow=False,
        yshift=14 if ll.loc[ll["lag"] == best, "r"].iloc[0] > 0 else -14,
        font={"color": t.ink, "size": 12},
    )
    fig.update_xaxes(title_text=f"Lag k ({unit}s) — k > 0: {driver} leads", zeroline=False)
    fig.update_yaxes(title_text="corr(driverₜ, targetₜ₊ₖ)", range=[-1.05, 1.05])
    _layout(
        fig, t, f"Lead / lag: {driver} → {target}", "Grey band = not significant at 5% (±1.96/√n)", height=360
    )
    fig.update_layout(hovermode="closest", showlegend=False)
    return fig


def cycle_chart(level: pd.Series, cyc: dict[str, Any], t: Theme, name: str, color: str) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.58, 0.42],
        vertical_spacing=0.08,
        subplot_titles=[f"{name} and HP trend", f"Cycle ({cyc['cycleUnit']})"],
    )
    fig.add_trace(
        go.Scatter(
            x=level.index,
            y=level,
            name=name,
            line={"color": color},
            hovertemplate="%{y:,.2f}<extra>" + name + "</extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=cyc["trend"].index,
            y=cyc["trend"],
            name="Trend",
            line={"color": t.ink_muted, "width": 1.5},
            hovertemplate="%{y:,.2f}<extra>Trend</extra>",
        ),
        row=1,
        col=1,
    )
    c = cyc["cycle"]
    fig.add_trace(
        go.Scatter(
            x=c.index,
            y=c,
            name="Cycle",
            line={"color": color},
            fill="tozeroy",
            fillcolor=_rgba(color, 0.12),
            hovertemplate="%{y:+.2f}<extra>Cycle</extra>",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    if cyc["peaks"]:
        fig.add_trace(
            go.Scatter(
                x=cyc["peaks"],
                y=c.loc[cyc["peaks"]],
                mode="markers",
                name="Peak",
                marker={
                    "symbol": "triangle-up",
                    "size": 9,
                    "color": t.ink,
                    "line": {"color": t.surface, "width": 2},
                },
                hovertemplate="Peak %{x|%b %Y}<extra></extra>",
            ),
            row=2,
            col=1,
        )
    if cyc["troughs"]:
        fig.add_trace(
            go.Scatter(
                x=cyc["troughs"],
                y=c.loc[cyc["troughs"]],
                mode="markers",
                name="Trough",
                marker={
                    "symbol": "triangle-down",
                    "size": 9,
                    "color": t.ink_muted,
                    "line": {"color": t.surface, "width": 2},
                },
                hovertemplate="Trough %{x|%b %Y}<extra></extra>",
            ),
            row=2,
            col=1,
        )
    fig.add_hline(y=0, line={"color": t.baseline, "width": 1}, row=2, col=1)
    for ann in fig.layout.annotations:
        ann.font = {"size": 12, "color": t.ink_secondary}
        ann.x, ann.xanchor = 0, "left"
    _layout(
        fig,
        t,
        f"Cycle decomposition — {name}",
        f"Phase: {cyc['phase']} ({cyc['phaseDescription']})",
        height=520,
    )
    fig.update_xaxes(showgrid=False, showline=True, linecolor=t.baseline)
    fig.update_yaxes(gridcolor=t.grid)
    return fig


def spectrum_chart(
    spectrum: pd.DataFrame, t: Theme, dominant: float | None, band: tuple[float, float]
) -> go.Figure:
    s = spectrum[(spectrum["periodYears"] <= 20) & (spectrum["periodYears"] >= 0.25)].sort_values(
        "periodYears"
    )
    fig = go.Figure()
    fig.add_vrect(x0=band[0], x1=band[1], fillcolor=_rgba(t.series[0], 0.07), line_width=0)
    fig.add_trace(
        go.Scatter(
            x=s["periodYears"],
            y=s["power"],
            mode="lines",
            line={"color": t.series[0]},
            name="Spectral power",
            hovertemplate="%{x:.1f}y · power %{y:.3g}<extra></extra>",
        )
    )
    if dominant:
        fig.add_vline(x=dominant, line={"color": t.ink_secondary, "width": 1})
        fig.add_annotation(
            x=dominant,
            y=float(s["power"].max()),
            text=f"dominant ≈ {dominant:.1f}y",
            showarrow=False,
            xanchor="left",
            xshift=6,
            font={"color": t.ink, "size": 12},
        )
    fig.update_xaxes(type="log", title_text="Cycle length (years, log scale)")
    fig.update_yaxes(title_text="Power")
    _layout(fig, t, "Periodogram of the cycle", "Shaded = business-cycle band searched", height=340)
    fig.update_layout(showlegend=False, hovermode="closest")
    return fig


def forecast_chart(level: pd.Series, fc: dict[str, Any], t: Theme, name: str, color: str) -> go.Figure:
    fig = go.Figure()
    f, lo, hi = fc["forecast"], fc["lower95"], fc["upper95"]
    bt = fc["backtest"]
    fig.add_trace(
        go.Scatter(
            x=level.index,
            y=level,
            name="Actual",
            line={"color": color},
            hovertemplate="%{y:,.2f}<extra>Actual</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=bt.index,
            y=bt["model"],
            name="Back-test (out-of-sample)",
            line={"color": t.ink_muted, "width": 1.5, "dash": "dot"},
            hovertemplate="%{y:,.2f}<extra>Back-test</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(hi.index) + list(lo.index[::-1]),
            y=list(hi) + list(lo[::-1]),
            fill="toself",
            fillcolor=_rgba(color, 0.14),
            line={"width": 0},
            name="95% band",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[level.index[-1], *f.index],
            y=[level.iloc[-1], *f],
            name="Forecast",
            mode="lines",
            line={"color": color, "dash": "dash"},
            hovertemplate="%{y:,.2f}<extra>Forecast</extra>",
        )
    )
    skill = fc["skill"]
    verdict = "beats naive" if fc["beatsNaive"] else "does NOT beat naive — illustrative only"
    sub = f"Out-of-sample skill vs random walk: {skill:+.0%} ({verdict})" if skill is not None else ""
    return _layout(fig, t, f"Forecast — {name}", sub, height=420)


def fitted_chart(y: pd.Series, fitted: pd.Series, t: Theme, name: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y.index,
            y=y,
            name="Actual",
            line={"color": t.series[0]},
            hovertemplate="%{y:,.2f}<extra>Actual</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fitted.index,
            y=fitted,
            name="Model fit",
            line={"color": t.series[1]},
            hovertemplate="%{y:,.2f}<extra>Fit</extra>",
        )
    )
    return _layout(fig, t, f"Regression fit — {name}", "Transformed basis", height=340)


# =========================================================================== Rate sensitivity


def rate_vs_benchmark_chart(
    panel: pd.DataFrame, industry: pd.DataFrame, bench_label: str, t: Theme
) -> go.Figure:
    """Highlight-one: each company a thin neutral line, industry median and
    benchmark emphasised, IQR band behind. All in % → one axis is valid."""
    fig = go.Figure()
    clean = panel[panel["effectiveRate"].notna()]
    for sym, g in clean.groupby("symbol"):
        g = g.sort_values("date")
        fig.add_trace(
            go.Scatter(
                x=g["date"].dt.year,
                y=g["effectiveRate"],
                mode="lines",
                line={"color": t.neutral_mark, "width": 1},
                name=str(g["company"].iloc[0]),
                showlegend=False,
                opacity=0.8,
                hovertemplate="%{y:.2f}%<extra>" + str(sym) + "</extra>",
            )
        )
    ind = industry.dropna(subset=["medianRate"])
    fig.add_trace(
        go.Scatter(
            x=list(ind.index) + list(ind.index[::-1]),
            y=list(ind["q3Rate"]) + list(ind["q1Rate"][::-1]),
            fill="toself",
            fillcolor=t.band,
            line={"width": 0},
            name="Interquartile range",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ind.index,
            y=ind["medianRate"],
            name="Industry median",
            mode="lines+markers",
            line={"color": t.series[0], "width": 2.5},
            marker={"size": 8, "line": {"color": t.surface, "width": 2}},
            hovertemplate="%{y:.2f}%<extra>Industry median</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ind.index,
            y=ind["benchmark"],
            name=bench_label,
            mode="lines+markers",
            line={"color": t.series[1], "width": 2.5},
            marker={"size": 8, "line": {"color": t.surface, "width": 2}},
            hovertemplate="%{y:.2f}%<extra>" + bench_label + "</extra>",
        )
    )
    fig.update_xaxes(title_text="Fiscal year", dtick=1)
    fig.update_yaxes(title_text="Percent", ticksuffix="%")
    return _layout(
        fig,
        t,
        "Effective cost of debt vs benchmark rate",
        "Grey lines = individual companies · interest expense ÷ average total debt",
        height=460,
    )


def spread_chart(industry: pd.DataFrame, bench_label: str, t: Theme) -> go.Figure:
    ind = industry.dropna(subset=["spread"])
    fig = go.Figure(
        go.Bar(
            x=ind.index,
            y=ind["spread"],
            marker={"color": t.series[0]},
            name="Spread",
            hovertemplate="<b>%{y:+.2f} pp</b><br>FY%{x}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line={"color": t.baseline, "width": 1})
    fig.update_xaxes(dtick=1, title_text="Fiscal year")
    fig.update_yaxes(title_text="Percentage points", ticksuffix=" pp")
    _layout(
        fig, t, "Credit spread over benchmark", f"Industry median effective rate − {bench_label}", height=340
    )
    fig.update_layout(hovermode="closest", showlegend=False)
    return fig


def coverage_chart(companies: pd.DataFrame, t: Theme) -> go.Figure:
    d = companies[companies["coverage"].notna() & ~companies["excluded"]].sort_values("coverage")
    d = d.assign(display=d["coverage"].clip(-30, 60))
    fig = go.Figure(
        go.Bar(
            y=d["company"],
            x=d["display"],
            orientation="h",
            marker={"color": t.series[0]},
            customdata=d["coverage"],
            name="Coverage",
            text=[f"{v:.1f}×" for v in d["coverage"]],
            textposition="outside",
            cliponaxis=False,
            textfont={"size": 12, "color": t.ink_secondary},
            hovertemplate="<b>%{customdata:.1f}×</b><br>%{y}<extra></extra>",
        )
    )
    for x, lbl, anchor in ((2, "2× thin ", "right"), (5, " 5× comfortable", "left")):
        fig.add_vline(x=x, line={"color": t.ink_muted, "width": 1})
        fig.add_annotation(
            x=x,
            y=1.0,
            yref="paper",
            text=lbl,
            showarrow=False,
            yanchor="bottom",
            xanchor=anchor,
            font={"size": 11, "color": t.ink_muted},
        )
    fig.add_vline(x=0, line={"color": t.baseline, "width": 1})
    fig.update_xaxes(
        title_text="EBIT ÷ interest expense (×, display clipped to −30…60)", showgrid=True, gridcolor=t.grid
    )
    fig.update_yaxes(showgrid=False)
    _layout(
        fig,
        t,
        "Interest coverage — latest fiscal year",
        "Negative = operating loss",
        height=max(320, 34 * len(d) + 140),
    )
    fig.update_layout(hovermode="closest", showlegend=False)
    return fig


def scenario_heatmap(scen: pd.DataFrame, t: Theme, beta: float) -> go.Figure:
    pv = scen.pivot_table(index="company", columns="shockBps", values="deltaInterestPctFull")
    pv = pv.reindex(pv.abs().max(axis=1).sort_values().index)
    lim = float(np.nanmax(np.abs(pv.to_numpy()))) if pv.size else 1.0
    neg, mid, pos = t.diverging
    fig = go.Figure(
        go.Heatmap(
            z=pv.to_numpy(),
            x=[f"{c:+d} bp" for c in pv.columns],
            y=list(pv.index),
            zmin=-lim,
            zmax=lim,
            zmid=0,
            colorscale=[[0, neg], [0.5, mid], [1, pos]],
            text=[[f"{v:+.0f}%" for v in row] for row in pv.to_numpy()],
            texttemplate="%{text}",
            textfont={"size": 11, "color": t.ink},
            xgap=2,
            ygap=2,
            hovertemplate="<b>%{z:+.1f}%</b> interest expense<br>%{y} · %{x}<extra></extra>",
            colorbar={
                "title": {"text": "Δ%", "font": {"color": t.ink_secondary}},
                "thickness": 10,
                "outlinewidth": 0,
                "tickfont": {"color": t.ink_muted},
            },
        )
    )
    fig.update_xaxes(showgrid=False, side="top")
    fig.update_yaxes(showgrid=False)
    _layout(
        fig,
        t,
        "Rate-shock scenarios: change in interest expense",
        f"Full repricing of average debt × β = {beta:.2f}",
        height=max(320, 36 * len(pv) + 160),
    )
    fig.update_layout(hovermode="closest", margin={"t": 120})
    return fig


# =========================================================================== Relative valuation


def pe_benchmark_chart(stock_pe: float | None, symbol: str, benchmarks: list[Any], t: Theme) -> go.Figure:
    rows = [(symbol, stock_pe, True)] + [(b.name, b.pe, False) for b in benchmarks if b.pe is not None]
    rows = [r for r in rows if r[1] is not None]
    names = [r[0] for r in rows][::-1]
    vals = [r[1] for r in rows][::-1]
    colors = [t.series[0] if r[2] else t.neutral_mark for r in rows][::-1]
    prem = {b.name: b.premium for b in benchmarks}
    labels = [
        f"{v:.1f}×" + ("" if n == symbol or prem.get(n) is None else f"  ({prem[n]:+.0%})")
        for n, v in zip(names, vals, strict=True)
    ]
    fig = go.Figure(
        go.Bar(
            y=names,
            x=vals,
            orientation="h",
            marker={"color": colors},
            text=labels,
            textposition="outside",
            textfont={"color": t.ink_secondary, "size": 12},
            cliponaxis=False,
            hovertemplate="<b>%{x:.1f}×</b><br>%{y}<extra></extra>",
        )
    )
    if stock_pe:
        fig.add_vline(x=stock_pe, line={"color": t.series[0], "width": 1})
    fig.update_xaxes(title_text="TTM P/E (×)", showgrid=True, gridcolor=t.grid, range=[0, max(vals) * 1.25])
    fig.update_yaxes(showgrid=False)
    _layout(
        fig,
        t,
        f"{symbol} P/E vs benchmarks",
        "(premium / discount of the stock vs each benchmark)",
        height=max(320, 40 * len(rows) + 140),
    )
    fig.update_layout(hovermode="closest", showlegend=False)
    return fig


def peer_scatter(peers: pd.DataFrame, stock: dict[str, Any], t: Theme, x: str, x_title: str) -> go.Figure:
    fig = go.Figure()
    d = peers[peers["peMeaningful"] & peers[x].notna()]
    for i, (set_name, g) in enumerate(d.groupby("set", sort=False)):
        fig.add_trace(
            go.Scatter(
                x=g[x] * 100,
                y=g["pe"],
                mode="markers+text",
                name=f"{set_name} peers",
                text=g["symbol"],
                textposition="top center",
                textfont={"size": 11, "color": t.ink_secondary},
                marker={
                    "size": 10,
                    "color": [t.series[0], t.series[2]][i % 2],
                    "line": {"color": t.surface, "width": 2},
                },
                customdata=g["company"],
                hovertemplate="<b>%{y:.1f}×</b> P/E<br>%{customdata}<br>"
                + x_title
                + " %{x:.1f}%<extra></extra>",
            )
        )
    if stock.get(x) is not None and stock.get("pe"):
        fig.add_trace(
            go.Scatter(
                x=[stock[x] * 100],
                y=[stock["pe"]],
                mode="markers+text",
                name=stock["symbol"],
                text=[f"<b>{stock['symbol']}</b>"],
                textposition="top center",
                textfont={"size": 12, "color": t.ink},
                marker={
                    "size": 14,
                    "color": t.series[1],
                    "symbol": "diamond",
                    "line": {"color": t.surface, "width": 2},
                },
                hovertemplate="<b>%{y:.1f}×</b> P/E<br>"
                + stock["symbol"]
                + "<br>"
                + x_title
                + " %{x:.1f}%<extra></extra>",
            )
        )
    fig.update_xaxes(
        title_text=f"{x_title} (%)", ticksuffix="%", showgrid=True, gridcolor=t.grid, zeroline=True
    )
    fig.update_yaxes(title_text="TTM P/E (×)")
    _layout(
        fig,
        t,
        f"Is the multiple earned? P/E vs {x_title.lower()}",
        "Meaningful P/Es only (0–200×)",
        height=460,
    )
    fig.update_layout(hovermode="closest")
    return fig


def pe_history_chart(
    history: pd.DataFrame, stats_: dict[str, Any], t: Theme, symbol: str, show_industry: bool
) -> go.Figure:
    fig = go.Figure()
    h = history
    if stats_:
        m, s = stats_["mean"], stats_["std"]
        fig.add_hrect(y0=m - s, y1=m + s, fillcolor=t.band, line_width=0)
        fig.add_hline(y=m, line={"color": t.series[0], "width": 1})
        fig.add_annotation(
            x=h.index[0],
            y=m + s,
            text="±1σ",
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            font={"size": 11, "color": t.ink_muted},
        )
        fig.add_annotation(
            x=h.index[0],
            y=m,
            text=f"avg {m:.1f}×",
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            font={"size": 11, "color": t.ink_muted},
        )
    fig.add_trace(
        go.Scatter(
            x=h.index,
            y=h["stockPE"],
            name=symbol,
            line={"color": t.series[0]},
            hovertemplate="%{y:.1f}×<extra>" + symbol + "</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=h.index,
            y=h["sectorPE"],
            name="Sector P/E (FMP)",
            line={"color": t.series[1]},
            hovertemplate="%{y:.1f}×<extra>Sector</extra>",
        )
    )
    if show_industry:
        fig.add_trace(
            go.Scatter(
                x=h.index,
                y=h["industryPE"],
                name="Industry P/E (FMP)",
                line={"color": t.series[2]},
                hovertemplate="%{y:.1f}×<extra>Industry</extra>",
            )
        )
    fig.update_yaxes(title_text="TTM P/E (×)", ticksuffix="×")
    return _layout(
        fig,
        t,
        f"{symbol} P/E through time",
        "Point-in-time TTM EPS (filing-date aligned); band = own mean ±1σ",
        height=440,
    )


def relative_pe_chart(history: pd.DataFrame, t: Theme, symbol: str) -> go.Figure:
    rel = history["relativeToSector"].dropna()
    fig = go.Figure()
    if not rel.empty:
        m, s = rel.mean(), rel.std(ddof=1)
        fig.add_hrect(y0=m - s, y1=m + s, fillcolor=t.band, line_width=0)
        fig.add_hline(y=m, line={"color": t.series[0], "width": 1})
    fig.add_hline(y=1, line={"color": t.baseline, "width": 1})
    fig.add_trace(
        go.Scatter(
            x=rel.index,
            y=rel,
            name="Relative P/E",
            line={"color": t.series[0]},
            hovertemplate="%{y:.2f}× sector<extra></extra>",
        )
    )
    fig.update_yaxes(title_text="Stock P/E ÷ sector P/E", ticksuffix="×")
    _layout(fig, t, f"{symbol} relative P/E vs sector", "1.0 = parity; band = mean ±1σ", height=340)
    fig.update_layout(showlegend=False)
    return fig


# =========================================================================== Fundamentals panel

_SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def _fmt_metric(v: float, fmt: str) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return f"{v:.0%}" if fmt == "pct" and abs(v) >= 0.1 else f"{v:.1%}" if fmt == "pct" else f"{v:.1f}×"


def sector_heatmap(
    med: pd.DataFrame, cnt: pd.DataFrame, t: Theme, label: str, fmt: str, diverging: bool
) -> go.Figure:
    """Sector × year median. Sequential (one hue) for levels; diverging
    (blue↔red through grey, centred at 0) for growth rates."""
    z = med.to_numpy(dtype=float)
    years = [str(c) for c in med.columns]
    n_np = cnt.reindex(index=med.index, columns=med.columns).fillna(0).to_numpy(dtype=float)
    text = [[_fmt_metric(v, fmt) for v in row] for row in z]
    hover = [
        [
            f"<b>{_fmt_metric(z[i, j], fmt) or 'n/a (too few companies)'}</b><br>{s} · {years[j]}<br>"
            f"{int(n_np[i, j])} companies"
            for j in range(len(years))
        ]
        for i, s in enumerate(med.index)
    ]
    if diverging:
        neg, mid, pos = t.diverging
        lim = float(np.nanmax(np.abs(z))) if np.isfinite(z).any() else 1.0
        kw: dict[str, Any] = {
            "colorscale": [[0, neg], [0.5, mid], [1, pos]],
            "zmid": 0,
            "zmin": -lim,
            "zmax": lim,
        }
    else:
        kw = {"colorscale": [[i / (len(_SEQ_BLUE) - 1), c] for i, c in enumerate(_SEQ_BLUE)]}
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=years,
            y=list(med.index),
            text=text,
            texttemplate="%{text}",
            textfont={"size": 11},
            hovertext=hover,
            hovertemplate="%{hovertext}<extra></extra>",
            xgap=2,
            ygap=2,
            colorbar={
                "thickness": 10,
                "outlinewidth": 0,
                "tickformat": ".0%" if fmt == "pct" else ".1f",
                "tickfont": {"color": t.ink_muted},
            },
            **kw,
        )
    )
    fig.update_xaxes(showgrid=False, side="top", type="category")
    fig.update_yaxes(showgrid=False, autorange="reversed")
    _layout(
        fig,
        t,
        f"{label} — median by sector and year",
        "Blank = fewer than 5 companies · hover for company counts",
        height=max(380, 30 * len(med) + 150),
    )
    fig.update_layout(hovermode="closest", margin={"t": 120})
    return fig


def sector_lines(med: pd.DataFrame, t: Theme, label: str, fmt: str, highlight: list[str]) -> go.Figure:
    """Highlight up to 6 chosen sectors in fixed colour order; others thin grey."""
    fig = go.Figure()
    years = [int(c) for c in med.columns]
    tick = ".0%" if fmt == "pct" else ".1f"
    for s in med.index:
        if s in highlight or s == "All companies":
            continue
        fig.add_trace(
            go.Scatter(
                x=years,
                y=med.loc[s],
                mode="lines",
                line={"color": t.neutral_mark, "width": 1},
                name=s,
                showlegend=False,
                hovertemplate="%{y:" + tick + "}<extra>" + s + "</extra>",
            )
        )
    for i, s in enumerate(highlight[:6]):
        if s in med.index:
            fig.add_trace(
                go.Scatter(
                    x=years,
                    y=med.loc[s],
                    mode="lines+markers",
                    name=s,
                    line={"color": series_color(t, i)},
                    marker={"size": 7, "line": {"color": t.surface, "width": 2}},
                    hovertemplate="%{y:" + tick + "}<extra>" + s + "</extra>",
                )
            )
    if "All companies" in med.index:
        fig.add_trace(
            go.Scatter(
                x=years,
                y=med.loc["All companies"],
                mode="lines",
                name="All companies",
                line={"color": t.ink, "width": 2, "dash": "dot"},
                hovertemplate="%{y:" + tick + "}<extra>All companies</extra>",
            )
        )
    fig.update_yaxes(tickformat=tick, title_text=label)
    fig.update_xaxes(dtick=1)
    return _layout(
        fig,
        t,
        f"{label} — sector medians over time",
        "Grey = other sectors · dotted = all companies",
        height=440,
    )


def leaderboard_scatter(lb: pd.DataFrame, t: Theme, highlight_sector: str | None) -> go.Figure:
    d = lb.dropna(subset=["revenueCAGR", "avgOperatingMargin"])
    if "smallBase" in d:
        d = d[~d["smallBase"]]  # CAGR from a tiny base is not meaningful
    fig = go.Figure()
    other = d[d["sector"] != highlight_sector] if highlight_sector else d
    fig.add_trace(
        go.Scatter(
            x=other["revenueCAGR"] * 100,
            y=other["avgOperatingMargin"] * 100,
            mode="markers",
            name="All companies" if not highlight_sector else "Other sectors",
            marker={
                "size": 8,
                "color": t.neutral_mark if highlight_sector else t.series[0],
                "opacity": 0.75,
                "line": {"color": t.surface, "width": 1},
            },
            customdata=np.stack([other["symbol"], other["company"], other["sector"]], axis=1),
            hovertemplate="<b>%{customdata[0]}</b> %{customdata[1]}<br>%{customdata[2]}<br>"
            "Revenue CAGR %{x:.1f}% · avg op. margin %{y:.1f}%<extra></extra>",
        )
    )
    if highlight_sector:
        h = d[d["sector"] == highlight_sector]
        # Selective direct labels: the 12 largest companies only (hover has the rest).
        biggest = set(h.nlargest(12, "marketCap")["symbol"]) if "marketCap" in h else set()
        fig.add_trace(
            go.Scatter(
                x=h["revenueCAGR"] * 100,
                y=h["avgOperatingMargin"] * 100,
                mode="markers+text",
                name=highlight_sector,
                text=[s if s in biggest else "" for s in h["symbol"]],
                textposition="top center",
                textfont={"size": 10, "color": t.ink_secondary},
                marker={"size": 10, "color": t.series[0], "line": {"color": t.surface, "width": 2}},
                customdata=np.stack([h["symbol"], h["company"], h["sector"]], axis=1),
                hovertemplate="<b>%{customdata[0]}</b> %{customdata[1]}<br>%{customdata[2]}<br>"
                "Revenue CAGR %{x:.1f}% · avg op. margin %{y:.1f}%<extra></extra>",
            )
        )
    fig.add_vline(x=0, line={"color": t.baseline, "width": 1})
    fig.add_hline(y=0, line={"color": t.baseline, "width": 1})
    fig.update_xaxes(title_text="Revenue CAGR (%)", ticksuffix="%", showgrid=True, gridcolor=t.grid)
    fig.update_yaxes(title_text="Average operating margin (%)", ticksuffix="%")
    _layout(
        fig,
        t,
        "Growth vs profitability",
        "Each dot is a company (small-base excluded) · labels = 12 largest in the highlighted sector",
        height=500,
    )
    fig.update_layout(hovermode="closest")
    return fig


def company_financials(hist: pd.DataFrame, t: Theme, symbol: str, currency: str) -> go.Figure:
    """Revenue and net income share one currency axis (same unit)."""
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=hist["year"],
            y=hist["revenue"] / 1e9,
            name="Revenue",
            marker={"color": t.series[0]},
            hovertemplate="%{y:,.1f}bn<extra>Revenue</extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=hist["year"],
            y=hist["netIncome"] / 1e9,
            name="Net income",
            marker={"color": t.series[1]},
            hovertemplate="%{y:,.1f}bn<extra>Net income</extra>",
        )
    )
    fig.add_hline(y=0, line={"color": t.baseline, "width": 1})
    fig.update_xaxes(dtick=1)
    fig.update_yaxes(title_text=f"{currency} billions")
    _layout(fig, t, f"{symbol} revenue and net income", f"Reported in {currency}", height=380)
    fig.update_layout(barmode="group", bargroupgap=0.08)
    return fig


def company_margins(hist: pd.DataFrame, t: Theme, symbol: str) -> go.Figure:
    fig = go.Figure()
    for i, (col, name) in enumerate(
        (("grossMargin", "Gross"), ("operatingMargin", "Operating"), ("netMargin", "Net"))
    ):
        fig.add_trace(
            go.Scatter(
                x=hist["year"],
                y=hist[col],
                name=f"{name} margin",
                mode="lines+markers",
                line={"color": series_color(t, i)},
                marker={"size": 7, "line": {"color": t.surface, "width": 2}},
                hovertemplate="%{y:.1%}<extra>" + name + "</extra>",
            )
        )
    fig.add_hline(y=0, line={"color": t.baseline, "width": 1})
    fig.update_xaxes(dtick=1)
    fig.update_yaxes(tickformat=".0%", title_text="Margin")
    return _layout(fig, t, f"{symbol} margins", "Gross · operating · net", height=360)


# =========================================================================== Stock analysis & peer comparison
#
# Colour follows the entity: the subject is always series[0], highlighted
# peers take series[1..] in the order chosen, every other peer is a thin grey
# line, the peer median is a dotted ink line and the market benchmark dashed.


def entity_colors(t: Theme, subject: str, highlight: list[str]) -> dict[str, str]:
    colors = {subject: t.series[0]}
    for i, s in enumerate([h for h in highlight if h != subject][: len(t.series) - 1]):
        colors[s] = t.series[i + 1]
    return colors


def _tick(fmt: str) -> dict[str, str]:
    from src.reporting.formatting import plotly_tick

    return plotly_tick(fmt)


def _hov(fmt: str, axis: str = "y") -> str:
    from src.reporting.formatting import hover

    return hover(fmt, axis)


def indicator_panels(
    panels: list[dict[str, Any]],
    t: Theme,
    subject: str,
    colors: dict[str, str],
    title: str,
    subtitle: str,
    cols: int = 2,
) -> go.Figure:
    """Small multiples, one panel per indicator, own y-scale, shared fiscal-year axis.

    Each panel dict: {"label", "fmt", "data": DataFrame(fiscalYear × symbol),
    "median": Series | None, "aggregate": Series | None}."""
    n = len(panels)
    cols = min(cols, n) or 1
    rows = int(np.ceil(n / cols)) or 1
    fig = make_subplots(
        rows=rows,
        cols=cols,
        shared_xaxes=False,
        subplot_titles=[p["label"] for p in panels],
        vertical_spacing=min(0.12, 0.5 / rows) if rows > 1 else 0.1,
        horizontal_spacing=0.09,
    )
    shown: set[str] = set()
    for i, p in enumerate(panels):
        r, c = i // cols + 1, i % cols + 1
        data: pd.DataFrame = p["data"]
        fmt = p["fmt"]
        years = [int(y) for y in data.index]
        order = [s for s in data.columns if s not in colors] + [s for s in colors if s in data.columns]
        for sym in order:
            y = data[sym]
            if y.notna().sum() == 0:
                continue
            col = colors.get(sym)
            emph = col is not None
            fig.add_trace(
                go.Scatter(
                    x=years,
                    y=y,
                    mode="lines+markers" if emph else "lines",
                    name=sym,
                    legendgroup=sym,
                    showlegend=emph and sym not in shown,
                    line={
                        "color": col or t.neutral_mark,
                        "width": 2.4 if sym == subject else (2 if emph else 1),
                    },
                    marker={"size": 6, "line": {"color": t.surface, "width": 1.5}} if emph else None,
                    connectgaps=False,
                    hovertemplate=_hov(fmt) + "<extra>" + sym + "</extra>",
                ),
                row=r,
                col=c,
            )
            if emph:
                shown.add(sym)
        for key, name, dash, color in (
            ("median", "Peer median", "dot", t.ink),
            ("aggregate", "Peer aggregate (Σ÷Σ)", "dash", t.ink_secondary),
        ):
            s = p.get(key)
            if s is not None and s.notna().any():
                fig.add_trace(
                    go.Scatter(
                        x=[int(y) for y in s.index],
                        y=s,
                        mode="lines",
                        name=name,
                        legendgroup=name,
                        showlegend=name not in shown,
                        line={"color": color, "width": 2, "dash": dash},
                        hovertemplate=_hov(fmt) + f"<extra>{name}</extra>",
                    ),
                    row=r,
                    col=c,
                )
                shown.add(name)
        fig.update_yaxes(**_tick(fmt), row=r, col=c)
        if fmt in ("pct",) and data.min(numeric_only=True).min() < 0:
            fig.add_hline(y=0, line={"color": t.baseline, "width": 1}, row=r, col=c)
    for ann in fig.layout.annotations:
        ann.font = {"size": 12, "color": t.ink_secondary}
    _layout(fig, t, title, subtitle, height=max(340, 250 * rows + 110))
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(dtick=1, showgrid=False, showline=True, linecolor=t.baseline)
    fig.update_yaxes(gridcolor=t.grid)
    return fig


def peer_rank_bar(
    values: pd.Series,
    t: Theme,
    subject: str,
    colors: dict[str, str],
    label: str,
    fmt: str,
    median: float | None,
    aggregate: float | None,
    higher_is_better: bool | None,
) -> go.Figure:
    """Horizontal bars, best at top (direction-adjusted); median / aggregate as reference lines."""
    v = values.dropna()
    v = v.sort_values(ascending=higher_is_better is False)
    fig = go.Figure(
        go.Bar(
            x=v.to_numpy(),
            y=list(v.index),
            orientation="h",
            marker={"color": [colors.get(s, t.neutral_mark) for s in v.index]},
            hovertemplate=_hov(fmt, "x") + "<extra>%{y}</extra>",
        )
    )
    for val, name, dash in ((median, "Peer median", "dot"), (aggregate, "Aggregate (Σ÷Σ)", "dash")):
        if val is not None and np.isfinite(val):
            fig.add_vline(x=val, line={"color": t.ink, "width": 1.5, "dash": dash})
            fig.add_annotation(
                x=val,
                y=1.0,
                yref="paper",
                text=name,
                showarrow=False,
                yanchor="bottom",
                font={"size": 11, "color": t.ink_secondary},
            )
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(**_tick(fmt))
    order = "best at top" if higher_is_better is not None else "sorted high → low"
    _layout(
        fig, t, label, f"Latest value · {order} · {subject} highlighted", height=max(300, 26 * len(v) + 140)
    )
    fig.update_layout(hovermode="closest", showlegend=False)
    return fig


def cumulative_return_chart(
    prices: pd.DataFrame, t: Theme, subject: str, colors: dict[str, str], bench: str | None
) -> go.Figure:
    """Price index rebased to 100 at the first common date — one axis, one unit."""
    p = prices.dropna(how="all")
    start = p.apply(lambda s: s.first_valid_index()).max()
    p = p.loc[start:].ffill()
    idx = p.divide(p.iloc[0]) * 100
    fig = go.Figure()
    order = [s for s in idx.columns if s not in colors and s != bench] + [
        s for s in [*([bench] if bench else []), *colors] if s in idx.columns
    ]
    for sym in order:
        col = colors.get(sym)
        is_b = sym == bench
        fig.add_trace(
            go.Scatter(
                x=idx.index,
                y=idx[sym],
                mode="lines",
                name=sym,
                showlegend=bool(col) or is_b,
                line={
                    "color": t.ink if is_b else (col or t.neutral_mark),
                    "width": 2 if (col or is_b) else 1,
                    "dash": "dash" if is_b else None,
                },
                hovertemplate="%{y:.0f}<extra>" + sym + "</extra>",
            )
        )
    fig.add_hline(y=100, line={"color": t.baseline, "width": 1})
    fig.update_yaxes(title_text=f"Index ({start:%b %Y} = 100)")
    return _layout(
        fig, t, "Cumulative price return", "Rebased to 100 · excludes dividends · grey = other peers", 440
    )


def rolling_vol_chart(prices: pd.DataFrame, t: Theme, colors: dict[str, str], bench: str | None) -> go.Figure:
    from src.analysis.indicators import rolling_volatility

    fig = go.Figure()
    order = [s for s in prices.columns if s not in colors and s != bench] + [
        s for s in [*([bench] if bench else []), *colors] if s in prices.columns
    ]
    for sym in order:
        vol = rolling_volatility(prices[sym].dropna())
        col = colors.get(sym)
        is_b = sym == bench
        fig.add_trace(
            go.Scatter(
                x=vol.index,
                y=vol,
                mode="lines",
                name=sym,
                showlegend=bool(col) or is_b,
                line={
                    "color": t.ink if is_b else (col or t.neutral_mark),
                    "width": 2 if (col or is_b) else 1,
                    "dash": "dash" if is_b else None,
                },
                hovertemplate="%{y:.1%}<extra>" + sym + "</extra>",
            )
        )
    fig.update_yaxes(tickformat=".0%", title_text="Annualised volatility")
    return _layout(fig, t, "Rolling 1-year volatility", "Std. dev. of daily log returns × √252", 400)


def drawdown_chart(prices: pd.DataFrame, t: Theme, colors: dict[str, str], bench: str | None) -> go.Figure:
    fig = go.Figure()
    for sym in [s for s in [*([bench] if bench else []), *colors] if s in prices.columns]:
        s = prices[sym].dropna()
        dd = s / s.cummax() - 1
        is_b = sym == bench
        fig.add_trace(
            go.Scatter(
                x=dd.index,
                y=dd,
                mode="lines",
                name=sym,
                fill=None if is_b else "tozeroy",
                fillcolor=None if is_b else _rgba(colors.get(sym, t.series[0]), 0.12),
                line={
                    "color": t.ink if is_b else colors[sym],
                    "width": 1.5,
                    "dash": "dash" if is_b else None,
                },
                hovertemplate="%{y:.1%}<extra>" + sym + "</extra>",
            )
        )
    fig.update_yaxes(tickformat=".0%", title_text="Below previous peak")
    return _layout(fig, t, "Drawdown", "Distance from the running price peak", 360)


def return_risk_scatter(
    tbl: pd.DataFrame, t: Theme, subject: str, colors: dict[str, str], ret_col: str, ret_label: str
) -> go.Figure:
    d = tbl.dropna(subset=["volatility", ret_col])
    fig = go.Figure()
    other = d[~d.index.isin(list(colors))]
    fig.add_trace(
        go.Scatter(
            x=other["volatility"],
            y=other[ret_col],
            mode="markers+text",
            text=list(other.index),
            textposition="top center",
            textfont={"size": 10, "color": t.ink_muted},
            name="Peers",
            marker={"size": 9, "color": t.neutral_mark, "line": {"color": t.surface, "width": 1}},
            hovertemplate="<b>%{text}</b><br>Volatility %{x:.1%} · return %{y:.1%}<extra></extra>",
        )
    )
    for sym, col in colors.items():
        if sym in d.index:
            fig.add_trace(
                go.Scatter(
                    x=[d.at[sym, "volatility"]],
                    y=[d.at[sym, ret_col]],
                    mode="markers+text",
                    text=[sym],
                    textposition="top center",
                    name=sym,
                    marker={
                        "size": 13 if sym == subject else 11,
                        "color": col,
                        "line": {"color": t.surface, "width": 2},
                    },
                    hovertemplate="<b>"
                    + sym
                    + "</b><br>Volatility %{x:.1%} · return %{y:.1%}<extra></extra>",
                )
            )
    fig.add_hline(y=0, line={"color": t.baseline, "width": 1})
    fig.update_xaxes(
        tickformat=".0%", title_text="Annualised volatility (risk)", showgrid=True, gridcolor=t.grid
    )
    fig.update_yaxes(tickformat=".0%", title_text=ret_label)
    _layout(fig, t, "Return vs risk", "Up-left is better: more return per unit of volatility", 460)
    fig.update_layout(hovermode="closest", showlegend=False)
    return fig


def price_ma_chart(prices: pd.Series, t: Theme, symbol: str, ccy: str | None) -> go.Figure:
    from src.reporting.formatting import ccy_symbol

    fig = go.Figure()
    sym = ccy_symbol(ccy)
    fig.add_trace(
        go.Scatter(
            x=prices.index,
            y=prices,
            mode="lines",
            name=symbol,
            line={"color": t.series[0], "width": 2},
            hovertemplate=sym + "%{y:,.2f}<extra>" + symbol + "</extra>",
        )
    )
    for w, color in ((50, t.series[1]), (200, t.series[2])):
        ma = prices.rolling(w, min_periods=w).mean()
        fig.add_trace(
            go.Scatter(
                x=ma.index,
                y=ma,
                mode="lines",
                name=f"{w}-day average",
                line={"color": color, "width": 1.5},
                hovertemplate=sym + "%{y:,.2f}<extra>" + f"{w}d avg</extra>",
            )
        )
    fig.update_yaxes(
        tickprefix=sym.strip() if len(sym) <= 3 else "",
        tickformat=",.0f",
        title_text=f"Price ({ccy or 'USD'})",
    )
    return _layout(fig, t, f"{symbol} share price", "Daily close with 50- and 200-day moving averages", 420)


def statement_bars(raw: pd.DataFrame, t: Theme, symbol: str, ccy: str | None) -> go.Figure:
    """Revenue, operating income, net income and free cash flow — one money axis."""
    from src.reporting.formatting import ccy_symbol

    sym = ccy_symbol(ccy)
    fig = go.Figure()
    years = [int(y) for y in raw.index]
    for i, (col, name) in enumerate(
        (
            ("inc_revenue", "Revenue"),
            ("inc_operatingIncome", "Operating income"),
            ("inc_netIncome", "Net income"),
            ("cf_freeCashFlow", "Free cash flow"),
        )
    ):
        if col not in raw:
            continue
        fig.add_trace(
            go.Bar(
                x=years,
                y=pd.to_numeric(raw[col], errors="coerce") / 1e9,
                name=name,
                marker={"color": series_color(t, i)},
                hovertemplate=sym + "%{y:,.1f}bn<extra>" + name + "</extra>",
            )
        )
    fig.add_hline(y=0, line={"color": t.baseline, "width": 1})
    fig.update_xaxes(dtick=1)
    fig.update_yaxes(
        title_text=f"{ccy or 'USD'} billions",
        tickprefix=sym.strip() if len(sym) <= 3 else "",
        ticksuffix="bn",
    )
    _layout(
        fig,
        t,
        f"{symbol} revenue, profit and free cash flow",
        f"Fiscal years · reported in {ccy or 'USD'}",
        420,
    )
    fig.update_layout(barmode="group", bargroupgap=0.06)
    return fig


# =========================================================================== DCF / Portfolio


def category_bars(
    values: pd.Series, t: Theme, title: str, subtitle: str | None = None, tickprefix: str = "$"
) -> go.Figure:
    """Vertical bars for a handful of labelled amounts (one entity per bar, colour follows it)."""
    v = values.dropna()
    fig = go.Figure(
        go.Bar(
            x=list(v.index),
            y=v.to_numpy(),
            marker={"color": [series_color(t, i) for i in range(len(v))]},
            hovertemplate=f"{tickprefix}%{{y:,.2f}}<extra>%{{x}}</extra>",
        )
    )
    fig.update_yaxes(tickprefix=tickprefix, tickformat="~s")
    _layout(fig, t, title, subtitle, height=360)
    fig.update_layout(hovermode="closest", showlegend=False)
    return fig


# =========================================================================== Backtest Lab


def equity_curve_chart(curves: pd.DataFrame, t: Theme, bench: str, colors: dict[str, str]) -> go.Figure:
    """Growth of $1 (log scale, one axis). Benchmark dashed ink; highlighted strategies coloured;
    other series grey."""
    fig = go.Figure()
    order = [c for c in curves.columns if c not in colors and c != bench] + [
        c for c in [bench, *colors] if c in curves.columns
    ]
    for name in order:
        col = colors.get(name)
        is_b = name == bench
        fig.add_trace(
            go.Scatter(
                x=curves.index,
                y=curves[name],
                mode="lines",
                name=name,
                showlegend=bool(col) or is_b,
                line={
                    "color": t.ink if is_b else (col or t.neutral_mark),
                    "width": 2 if (col or is_b) else 1,
                    "dash": "dash" if is_b else None,
                },
                hovertemplate="$%{y:.2f}<extra>" + name + "</extra>",
            )
        )
    fig.update_yaxes(type="log", title_text="Growth of $1 (log scale)")
    return _layout(
        fig, t, "Growth of $1 — net of costs", "Monthly rebalancing · total return · log scale", 460
    )


def walk_forward_chart(wf: pd.DataFrame, t: Theme, bench: str) -> go.Figure:
    """In-sample vs out-of-sample Sharpe per strategy (grouped bars, one axis)."""
    d = wf[["sharpeIS", "sharpeOOS"]].dropna(how="all").sort_values("sharpeIS", ascending=False)
    fig = go.Figure()
    for col, label, color in (
        ("sharpeIS", "In-sample", t.series[0]),
        ("sharpeOOS", "Out-of-sample", t.series[1]),
    ):
        fig.add_trace(
            go.Bar(
                x=list(d.index),
                y=d[col],
                name=label,
                marker={"color": color},
                hovertemplate="%{y:.2f}<extra>" + label + "</extra>",
            )
        )
    if bench in d.index:
        fig.add_hline(
            y=float(pd.to_numeric(d.loc[bench, "sharpeOOS"])),
            line={"color": t.ink, "width": 1.5, "dash": "dash"},
            annotation_text=f"{bench.upper()} out-of-sample",
            annotation_position="top right",
        )
    fig.update_yaxes(title_text="Sharpe ratio (annualised)")
    _layout(
        fig, t, "Walk-forward: does in-sample performance survive?", "Sharpe before vs after the split", 420
    )
    fig.update_layout(barmode="group", hovermode="x unified")
    return fig


def portfolio_value_chart(
    value: pd.Series, bench: pd.Series, t: Theme, name: str, bench_name: str
) -> go.Figure:
    """Simulated portfolio value in dollars vs the same capital in the benchmark (one axis)."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=bench.index,
            y=bench,
            mode="lines",
            name=bench_name,
            line={"color": t.ink, "width": 2, "dash": "dash"},
            hovertemplate="$%{y:,.0f}<extra>" + bench_name + "</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=value.index,
            y=value,
            mode="lines",
            name=name,
            line={"color": t.series[0], "width": 2},
            hovertemplate="$%{y:,.0f}<extra>" + name + "</extra>",
        )
    )
    fig.update_yaxes(title_text="Portfolio value ($)", tickprefix="$", tickformat=",.0f")
    return _layout(fig, t, "Portfolio value", "Net of transaction costs · dividends reinvested", 440)


def allocation_area_chart(weights: pd.DataFrame, t: Theme, title: str) -> go.Figure:
    """Stacked weights over time (dates x categories, rows sum to 1). Colour follows the
    category; largest average weight first."""
    order = list(weights.mean().sort_values(ascending=False).index)
    fig = go.Figure()
    for i, c in enumerate(order):
        color = t.neutral_mark if c == "Cash" else series_color(t, i)
        fig.add_trace(
            go.Scatter(
                x=weights.index,
                y=weights[c],
                mode="lines",
                name=str(c),
                stackgroup="one",
                line={"width": 0.5, "color": color},
                fillcolor=_rgba(color, 0.75),
                hovertemplate="%{y:.1%}<extra>" + str(c) + "</extra>",
            )
        )
    fig.update_yaxes(title_text="Portfolio weight", tickformat=".0%", range=[0, 1])
    _layout(fig, t, title, "At each rebalance, after trading", 420)
    fig.update_layout(hovermode="x unified")
    return fig
