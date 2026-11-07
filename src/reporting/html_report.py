# src/reporting/html_report.py
#
# Self-contained HTML research report (opens offline, prints cleanly, can be
# emailed). Same figures as the dashboard. Structure follows sell-side
# convention: header → key metrics → key findings → analysis sections →
# methodology → caveats & data quality → sources → disclaimer.

import html
import re
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go

from src.reporting.theme import FONT_STACK, LIGHT


@dataclass
class Kpi:
    label: str
    value: str
    note: str = ""


@dataclass
class Section:
    heading: str
    intro: str = ""
    figures: list[go.Figure] = field(default_factory=list)
    tables: list[tuple[str, pd.DataFrame]] = field(default_factory=list)


def md_inline(text: str) -> str:
    """Escape, then render **bold** — data (company names etc.) is never trusted HTML."""
    escaped = html.escape(text)
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)


def build_html_report(
    title: str,
    subtitle: str,
    kpis: list[Kpi],
    findings: list[str],
    sections: list[Section],
    methodology: list[str],
    caveats: list[str],
    sources: list[str],
) -> str:
    t = LIGHT
    first_fig = True
    body: list[str] = []
    for sec in sections:
        body.append(f"<section><h2>{html.escape(sec.heading)}</h2>")
        if sec.intro:
            body.append(f"<p class='intro'>{md_inline(sec.intro)}</p>")
        for fig in sec.figures:
            fig_html = fig.to_html(
                full_html=False,
                include_plotlyjs=first_fig,
                config={"displaylogo": False, "responsive": True},
            )
            first_fig = False
            body.append(f"<div class='figure'>{fig_html}</div>")
        for caption, df in sec.tables:
            body.append(f"<h3>{html.escape(caption)}</h3>")
            body.append(df.to_html(classes="data", border=0, na_rep="—", escape=True))
        body.append("</section>")

    kpi_html = "".join(
        f"<div class='kpi'><div class='kpi-label'>{html.escape(k.label)}</div>"
        f"<div class='kpi-value'>{html.escape(k.value)}</div>"
        f"<div class='kpi-note'>{md_inline(k.note)}</div></div>"
        for k in kpis
    )
    li = lambda items: "".join(f"<li>{md_inline(i)}</li>" for i in items)  # noqa: E731

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{ --surface:{t.surface}; --page:{t.page}; --ink:{t.ink}; --ink2:{t.ink_secondary};
  --muted:{t.ink_muted}; --grid:{t.grid}; --border:{t.border}; --accent:{t.series[0]}; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--page); color:var(--ink); font-family:{FONT_STACK}; line-height:1.5; }}
.wrap {{ max-width:1120px; margin:0 auto; padding:32px 24px 64px; }}
header {{ border-bottom:1px solid var(--grid); padding-bottom:20px; margin-bottom:24px; }}
.eyebrow {{ color:var(--muted); font-size:12px; letter-spacing:.06em; text-transform:uppercase; }}
h1 {{ font-size:28px; margin:6px 0 4px; font-weight:650; }}
.sub {{ color:var(--ink2); margin:0; }}
h2 {{ font-size:19px; margin:36px 0 12px; font-weight:620; }}
h3 {{ font-size:14px; color:var(--ink2); margin:20px 0 8px; font-weight:600; }}
.kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; }}
.kpi {{ background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:14px 16px; }}
.kpi-label {{ color:var(--muted); font-size:12px; }}
.kpi-value {{ font-size:24px; font-weight:620; margin-top:2px; }}
.kpi-note {{ color:var(--ink2); font-size:12px; }}
.card {{ background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:16px 20px; }}
.card ul {{ margin:0; padding-left:20px; }} .card li {{ margin:6px 0; }}
.figure {{ background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:8px; margin:12px 0; }}
.intro {{ color:var(--ink2); }}
table.data {{ width:100%; border-collapse:collapse; font-size:13px; background:var(--surface);
  border:1px solid var(--border); border-radius:10px; overflow:hidden; font-variant-numeric:tabular-nums; }}
table.data th {{ text-align:left; color:var(--ink2); font-weight:600; border-bottom:1px solid var(--grid); padding:8px 10px; }}
table.data td {{ border-bottom:1px solid var(--grid); padding:7px 10px; }}
.caveats li {{ color:var(--ink2); }}
footer {{ margin-top:40px; color:var(--muted); font-size:12px; border-top:1px solid var(--grid); padding-top:16px; }}
@media print {{ .figure, .kpi, .card {{ break-inside:avoid; }} body {{ background:#fff; }} }}
</style></head>
<body><div class="wrap">
<header><div class="eyebrow">Research report · generated {datetime.now():%d %b %Y %H:%M}</div>
<h1>{html.escape(title)}</h1><p class="sub">{md_inline(subtitle)}</p></header>
<div class="kpis">{kpi_html}</div>
<h2>Key findings</h2><div class="card"><ul>{li(findings)}</ul></div>
{"".join(body)}
<h2>Methodology</h2><div class="card"><ol>{li(methodology)}</ol></div>
<h2>Caveats &amp; data quality</h2><div class="card caveats"><ul>{li(caveats) or "<li>None flagged.</li>"}</ul></div>
<h2>Data sources</h2><div class="card"><ul>{li(sources)}</ul></div>
<footer>For personal research use. Not investment advice. Statistical relationships are historical and may not
persist; forecasts are shown with their out-of-sample skill and should be read alongside it.</footer>
</div></body></html>"""
