# src/dashboard/ui.py
#
# Shared Streamlit UI kit: page chrome, KPI tiles, findings / methodology /
# caveat blocks, themed charts and report download. Every page uses these so
# the app looks and behaves as one product. All user/API-derived text is
# HTML-escaped before it reaches unsafe_allow_html.

import contextlib
import html
import re
from collections.abc import Iterable
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.reporting.html_report import Kpi
from src.reporting.theme import Theme, get_theme


def current_theme() -> Theme:
    try:
        mode = st.context.theme.type  # "light" | "dark" (Streamlit >= 1.46)
    except AttributeError:
        mode = "light"
    return get_theme(mode)


def _md_inline(text: str) -> str:
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html.escape(text))


def inject_css(t: Theme) -> None:
    st.markdown(
        f"""
<style>
.block-container {{ padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1280px; }}
h1, h2, h3 {{ letter-spacing: -0.01em; }}
.rp-eyebrow {{ color: {t.ink_muted}; font-size: 12px; letter-spacing: .08em; text-transform: uppercase; margin-bottom: 2px; }}
.rp-title {{ font-size: 30px; font-weight: 650; margin: 0 0 4px 0; line-height: 1.2; }}
.rp-desc {{ color: {t.ink_secondary}; margin: 0 0 18px 0; max-width: 880px; }}
.rp-kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin: 6px 0 18px; }}
.rp-kpi {{ background: {t.surface}; border: 1px solid {t.border}; border-radius: 12px; padding: 14px 16px; }}
.rp-kpi-label {{ color: {t.ink_muted}; font-size: 12px; }}
.rp-kpi-value {{ color: {t.ink}; font-size: 24px; font-weight: 620; margin-top: 2px; line-height: 1.25; }}
.rp-kpi-note {{ color: {t.ink_secondary}; font-size: 12px; margin-top: 2px; }}
.rp-card {{ background: {t.surface}; border: 1px solid {t.border}; border-radius: 12px; padding: 14px 20px; margin: 6px 0 14px; }}
.rp-card ul, .rp-card ol {{ margin: 0; padding-left: 20px; }}
.rp-card li {{ margin: 6px 0; color: {t.ink}; }}
.rp-card.caution li {{ color: {t.ink_secondary}; }}
.rp-tag {{ display: inline-block; font-size: 11px; padding: 1px 8px; border-radius: 999px; border: 1px solid {t.border};
  color: {t.ink_secondary}; margin-right: 6px; }}
div[data-testid="stPlotlyChart"] {{ border: 1px solid {t.border}; border-radius: 12px; overflow: hidden; background: {t.surface}; }}
div[data-testid="stTabs"] button p {{ font-size: 14px; }}
</style>""",
        unsafe_allow_html=True,
    )


def setup_page(title: str, icon: str = "📈") -> Theme:
    """Call first on every page. Returns the active chart theme."""
    with contextlib.suppress(st.errors.StreamlitAPIException):  # already set by the router
        st.set_page_config(page_title=title, page_icon=icon, layout="wide")
    t = current_theme()
    inject_css(t)
    return t


def page_header(eyebrow: str, title: str, description: str) -> None:
    st.markdown(
        f"<div class='rp-eyebrow'>{html.escape(eyebrow)}</div>"
        f"<div class='rp-title'>{html.escape(title)}</div>"
        f"<p class='rp-desc'>{_md_inline(description)}</p>",
        unsafe_allow_html=True,
    )


def kpi_row(kpis: Iterable[Kpi]) -> None:
    cards = "".join(
        f"<div class='rp-kpi'><div class='rp-kpi-label'>{html.escape(k.label)}</div>"
        f"<div class='rp-kpi-value'>{html.escape(k.value)}</div>"
        f"<div class='rp-kpi-note'>{_md_inline(k.note)}</div></div>"
        for k in kpis
    )
    st.markdown(f"<div class='rp-kpis'>{cards}</div>", unsafe_allow_html=True)


def bullet_card(
    items: Iterable[str], ordered: bool = False, caution: bool = False, empty: str = "None."
) -> None:
    items = list(items)
    tag = "ol" if ordered else "ul"
    body = "".join(f"<li>{_md_inline(i)}</li>" for i in items) or f"<li>{html.escape(empty)}</li>"
    st.markdown(
        f"<div class='rp-card{' caution' if caution else ''}'><{tag}>{body}</{tag}></div>",
        unsafe_allow_html=True,
    )


def findings(items: Iterable[str]) -> None:
    st.markdown("#### Key findings")
    bullet_card(items, empty="No findings.")


def report_footer(methodology: list[str], caveats: list[str], sources: list[str]) -> None:
    """Methodology, caveats and sources — always present, never hidden by default for caveats."""
    if caveats:
        st.markdown("#### Caveats & data quality")
        bullet_card(caveats, caution=True)
    with st.expander("Methodology", expanded=False):
        bullet_card(methodology, ordered=True)
    with st.expander("Data sources", expanded=False):
        bullet_card(sources)


def chart(fig: go.Figure, key: str | None = None) -> None:
    st.plotly_chart(fig, width="stretch", theme=None, key=key, config={"displaylogo": False})


def table(df: pd.DataFrame, **kwargs: Any) -> None:
    st.dataframe(df, width="stretch", **kwargs)


def download_report(html_doc: str, file_name: str, label: str = "Download full report (HTML)") -> None:
    st.download_button(
        label, data=html_doc.encode("utf-8"), file_name=file_name, mime="text/html", type="secondary"
    )


def empty_state(message: str) -> None:
    st.markdown(
        f"<div class='rp-card'><span class='rp-tag'>Ready</span>{html.escape(message)}</div>",
        unsafe_allow_html=True,
    )
