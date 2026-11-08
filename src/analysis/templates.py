# src/analysis/templates.py
#
# Industry scoring templates — the single answer to "which indicators judge this company?".
# A template (config.SCORECARD_TEMPLATES) names, per scorecard dimension, the metrics that are
# scored, plus dimensions unscored by design, display indicators, valuation methods and the
# industry metrics FMP Starter does not provide. Metric values are computed here from PLAIN
# statement dicts (FMP field names), so the live scorecard (stock_analysis) and the point-in-time
# backtest (backtest/scorecard) run the same code. Pure: no I/O.

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.analysis.business_model import revenue_pair, scorecard_revenue, yoy_growth
from src.config import SCORECARD_METRIC_BANDS, SCORECARD_TEMPLATES

# metric key -> (label, format) ; format "pct" | "x" (multiple) | "ratio"
METRICS: dict[str, tuple[str, str]] = {
    "roe": ("ROE", "pct"),
    "netMargin": ("net margin", "pct"),
    "revenueGrowth": ("revenue growth", "pct"),
    "netIncomeGrowth": ("net income growth", "pct"),
    "epsGrowth": ("EPS growth", "pct"),
    "currentRatio": ("current ratio", "ratio"),
    "debtToEquity": ("debt/equity", "ratio"),
    "interestCoverage": ("interest coverage", "x"),
    "equityToAssets": ("equity/assets", "pct"),
    "roa": ("ROA", "pct"),
    "efficiencyRatio": ("efficiency ratio", "pct"),
}


@dataclass(frozen=True)
class Template:
    key: str
    label: str
    dimensions: dict[str, list[str]]  # dimension -> scored metric keys
    unscored: dict[str, str]  # dimension -> reason (unscored by design)
    why: str
    valuation: list[str]
    display: list[str]
    unavailable: list[str]


@dataclass
class MetricSet:
    values: dict[str, float | None] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)  # metric -> why it is n/m (shown in rationale)
    revenue_basis: str = "reported"  # "reported" | "net of interest expense" | "changed between years"


def _tpl(key: str) -> Template:
    t = SCORECARD_TEMPLATES[key]
    return Template(
        key=key,
        label=t["label"],
        dimensions=t["dimensions"],
        unscored=t.get("unscored", {}),
        why=t["why"],
        valuation=t["valuation"],
        display=t["display"],
        unavailable=t["unavailable"],
    )


def template_for(kind: str, captive_finance: bool = False) -> Template:
    """The template for a business model. Captive-finance groups use the general template with
    financial health unscored (consolidated ratios mix in the finance arm)."""
    key = next((k for k, t in SCORECARD_TEMPLATES.items() if kind in t["kinds"]), "standard")
    t = _tpl(key)
    if captive_finance and "financialHealth" in t.dimensions:
        dims = {k: v for k, v in t.dimensions.items() if k != "financialHealth"}
        unscored = {
            **t.unscored,
            "financialHealth": "not scored — captive-finance arm makes consolidated current ratio, D/E and "
            "coverage not meaningful for the operating business",
        }
        t = Template(t.key, t.label, dims, unscored, t.why, t.valuation, t.display, t.unavailable)
    return t


def _f(v: Any) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _ratio(a: Any, b: Any, positive_denominator: bool = True) -> float | None:
    x, y = _f(a), _f(b)
    if x is None or y is None or y == 0 or (positive_denominator and y < 0):
        return None
    return x / y


def compute_metrics(
    income: dict[str, Any] | None,
    prior_income: dict[str, Any] | None,
    balance: dict[str, Any] | None,
    cashflow: dict[str, Any] | None,
    prior_cashflow: dict[str, Any] | None,
    kind: str,
    subtype: str | None = None,
    captive_finance: bool = False,
) -> MetricSet:
    """Every template metric for one company-year, from FMP-named statement dicts. A metric that
    is not meaningful is None with a note; one that is missing is just None."""
    inc, pri, bal = income or {}, prior_income or {}, balance or {}
    # cashflow / prior_cashflow: part of the interface for templates that need cash-flow items
    # (none of the active templates do yet — the REIT FFO proxy was withdrawn)
    del cashflow, prior_cashflow
    m = MetricSet()
    v = m.values

    ni, equity, assets = (
        _f(inc.get("netIncome")),
        _f(bal.get("totalStockholdersEquity")),
        _f(bal.get("totalAssets")),
    )
    if equity == 0:
        equity = None  # zero equity is a data gap, not negative equity
    negative_equity = equity is not None and equity < 0

    # profitability
    v["roe"] = None if negative_equity else _ratio(ni, equity)
    if negative_equity:
        m.notes["roe"] = "n/m (negative equity)"
    rev_now, rev_prior, basis = revenue_pair(inc or None, pri or None, kind, subtype)
    rev_reported = _f(inc.get("revenue"))
    if basis == "mixed":
        m.revenue_basis = "changed between years"
    elif rev_now is not None and rev_reported is not None and rev_now != rev_reported:
        m.revenue_basis = "net of interest expense"
    v["netMargin"] = _ratio(ni, rev_now)
    v["roa"] = _ratio(ni, assets)
    # Efficiency ratio needs NET revenue: only when FMP presents this year gross (so revenue −
    # interest expense is known). Otherwise FMP's revenue is neither gross nor net (BKU: revenue
    # below interest income) and opex ÷ revenue would read 33% instead of ~53% — n/m.
    net_rev = scorecard_revenue(inc or None, kind, subtype)
    if net_rev is not None and rev_reported is not None and net_rev != rev_reported:
        v["efficiencyRatio"] = _ratio(inc.get("operatingExpenses"), net_rev)
    else:
        v["efficiencyRatio"] = None
        if kind == "bank":
            m.notes["efficiencyRatio"] = "n/m (FMP revenue is not presented gross, net revenue unknown)"

    # growth
    v["revenueGrowth"] = yoy_growth(_f(rev_now), _f(rev_prior))
    if basis == "mixed":
        m.notes["revenueGrowth"] = "n/m (FMP switched between gross and net presentation)"
    prior_ni = _f(pri.get("netIncome"))
    v["netIncomeGrowth"] = yoy_growth(ni, prior_ni)
    if v["netIncomeGrowth"] is None and prior_ni is not None and prior_ni <= 0 and ni is not None:
        m.notes["netIncomeGrowth"] = "n/m (prior-year loss)"
    eps, prior_eps = _f(inc.get("epsDiluted")), _f(pri.get("epsDiluted"))
    v["epsGrowth"] = yoy_growth(eps, prior_eps)
    if v["epsGrowth"] is None and prior_eps is not None and prior_eps <= 0 and eps is not None:
        m.notes["epsGrowth"] = "n/m (prior-year loss)"

    # financial health (general)
    v["currentRatio"] = _ratio(bal.get("totalCurrentAssets"), bal.get("totalCurrentLiabilities"))
    if negative_equity:
        v["debtToEquity"] = None
        m.notes["debtToEquity"] = "n/m (negative equity)"
    elif captive_finance:
        v["debtToEquity"] = None
        m.notes["debtToEquity"] = "n/m (includes captive-finance debt)"
    else:
        v["debtToEquity"] = _ratio(bal.get("totalDebt"), equity, positive_denominator=False)
    interest = _f(inc.get("interestExpense"))
    if captive_finance:
        v["interestCoverage"] = None
        m.notes["interestCoverage"] = "n/m (finance-arm interest booked in cost of sales)"
    elif not interest or interest <= 0:
        # FMP reports 0 when interest is netted (AAPL) — a data gap, not zero debt service
        v["interestCoverage"] = None
        if interest == 0:
            m.notes["interestCoverage"] = "data unreliable (near-zero reported interest expense)"
    else:
        v["interestCoverage"] = _ratio(inc.get("operatingIncome"), interest)
    v["equityToAssets"] = _ratio(equity, assets)

    return m


def _band(value: float | None, metric: str) -> int | None:
    if value is None:
        return None
    spec = SCORECARD_METRIC_BANDS[metric]
    bands: list[tuple[float, int]] = spec["bands"]
    for threshold, score in bands:
        if (value >= threshold) if spec["mode"] == "higher" else (value <= threshold):
            return score
    return bands[-1][1]


def _fmt(metric: str, value: float) -> str:
    label, fmt = METRICS[metric]
    if fmt == "pct":
        return f"{label} {value:.1%}"
    if fmt == "x":
        return f"{label} {value:.1f}x"
    return f"{label} {value:.2f}"


def score_dimensions(t: Template, m: MetricSet) -> dict[str, dict[str, Any]]:
    """{dimension: {score, rationale, detail}} for the template's business dimensions
    (momentum is scored separately, the same for every template)."""
    out: dict[str, dict[str, Any]] = {}
    for dim, reason in t.unscored.items():
        out[dim] = {"score": None, "rationale": reason, "detail": {}}
    for dim, metrics in t.dimensions.items():
        scores, notes = [], []
        if dim == "growth" and m.revenue_basis == "net of interest expense" and "revenueGrowth" in metrics:
            notes.append("revenue net of interest expense (FMP reports it gross)")
        for k in metrics:
            val = m.values.get(k)
            s = _band(val, k)
            if s is not None:
                scores.append(s)
            if val is not None:
                notes.append(_fmt(k, val))
            elif k in m.notes:
                notes.append(f"{METRICS[k][0]} {m.notes[k]}")
        out[dim] = {
            "score": round(sum(scores) / len(scores)) if scores else None,
            "rationale": ", ".join(notes) if notes else "Insufficient data",
            "detail": {k: m.values.get(k) for k in metrics},
            "metrics": metrics,
        }
    return out


def describe(t: Template) -> dict[str, Any]:
    """Plain description for reports, the UI and agent prompts."""
    return {
        "template": t.label,
        "scored": {d: [METRICS[k][0] for k in ks] for d, ks in t.dimensions.items()},
        "unscored": t.unscored,
        "why": t.why,
        "valuation": t.valuation,
        "unavailable": t.unavailable,
        "display": t.display,
    }
