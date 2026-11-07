# src/reporting/theme.py
#
# Design tokens for every chart and report surface (dashboard + HTML export).
# Categorical order is fixed (validated with the dataviz skill's palette
# validator: first 3 slots pass all-pairs CVD checks in both modes; up to 6
# pass for adjacent series). Colour follows the entity, never its rank.

from dataclasses import dataclass

import plotly.graph_objects as go
import plotly.io as pio

FONT_STACK = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'


@dataclass(frozen=True)
class Theme:
    mode: str
    surface: str
    page: str
    ink: str
    ink_secondary: str
    ink_muted: str
    grid: str
    baseline: str
    border: str
    series: tuple[str, ...]
    diverging: tuple[str, str, str]  # negative pole, neutral midpoint, positive pole
    neutral_mark: str  # de-emphasised series ("the rest" in highlight charts)
    band: str  # confidence / IQR band fill
    good: str = "#0ca30c"
    warning: str = "#fab219"
    serious: str = "#ec835a"
    critical: str = "#d03b3b"


LIGHT = Theme(
    mode="light",
    surface="#fcfcfb",
    page="#f9f9f7",
    ink="#0b0b0b",
    ink_secondary="#52514e",
    ink_muted="#898781",
    grid="#e1e0d9",
    baseline="#c3c2b7",
    border="rgba(11,11,11,0.10)",
    series=("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"),
    diverging=("#2a78d6", "#f0efec", "#e34948"),
    neutral_mark="#c3c2b7",
    band="rgba(42,120,214,0.14)",
)

DARK = Theme(
    mode="dark",
    surface="#1a1a19",
    page="#0d0d0d",
    ink="#f0efec",
    ink_secondary="#c3c2b7",
    ink_muted="#898781",
    grid="#2c2c2a",
    baseline="#383835",
    border="rgba(255,255,255,0.10)",
    series=("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"),
    diverging=("#3987e5", "#383835", "#e66767"),
    neutral_mark="#52514e",
    band="rgba(57,135,229,0.20)",
)


def get_theme(mode: str | None) -> Theme:
    return DARK if mode == "dark" else LIGHT


def plotly_template(t: Theme) -> go.layout.Template:
    axis = {
        "gridcolor": t.grid,
        "gridwidth": 1,
        "linecolor": t.baseline,
        "zerolinecolor": t.baseline,
        "zerolinewidth": 1,
        "tickfont": {"color": t.ink_muted, "size": 12},
        "title": {"font": {"color": t.ink_secondary, "size": 12}},
        "ticks": "",
        "showline": False,
        "automargin": True,  # long category labels (company names) never clip
    }
    return go.layout.Template(
        layout={
            "font": {"family": FONT_STACK, "color": t.ink, "size": 13},
            "paper_bgcolor": t.surface,
            "plot_bgcolor": t.surface,
            "colorway": list(t.series),
            "title": {"font": {"size": 15, "color": t.ink}, "x": 0, "xanchor": "left", "y": 0.97},
            "xaxis": {**axis, "showgrid": False, "showline": True},
            "yaxis": {**axis, "showgrid": True},
            "legend": {
                "orientation": "h",
                "x": 0,
                "xanchor": "left",
                "y": 1.02,
                "yanchor": "bottom",
                "font": {"color": t.ink_secondary, "size": 12},
                "bgcolor": "rgba(0,0,0,0)",
            },
            "hoverlabel": {
                "bgcolor": t.surface,
                "bordercolor": t.baseline,
                "font": {"family": FONT_STACK, "color": t.ink, "size": 12},
            },
            "hovermode": "x unified",
            "margin": {"l": 16, "r": 24, "t": 64, "b": 40},
            "bargap": 0.25,
        },
        data={
            "scatter": [go.Scatter(line={"width": 2})],
            "bar": [go.Bar(marker={"line": {"width": 0}})],
        },
    )


def register_templates() -> None:
    pio.templates["research_light"] = plotly_template(LIGHT)
    pio.templates["research_dark"] = plotly_template(DARK)


register_templates()
