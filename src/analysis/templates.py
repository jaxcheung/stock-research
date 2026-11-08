# src/analysis/templates.py
#
# Industry scoring templates — the single answer to "which indicators judge this company?".
# A template (config.SCORECARD_TEMPLATES) names, per scorecard dimension, the metrics that are
# scored, plus dimensions unscored by design, display indicators, valuation methods and the
# industry metrics FMP Starter does not provide. Metric values are computed here from PLAIN
# statement dicts (FMP field names), so the live scorecard (stock_analysis) and the point-in-time
# backtest (backtest/scorecard) run the same code. Pure: no I/O.

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.analysis.business_model import financing_debt, revenue_pair, scorecard_revenue, yoy_growth
from src.config import (
    REIT_CFO_JUMP,
    REIT_DIVIDEND_JUMP,
    SCORECARD_METRIC_BANDS,
    SCORECARD_TEMPLATES,
)

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
    "cfoDividendCoverage": ("dividend coverage (operating cash flow ÷ dividends)", "x"),
    "cfoPerShareGrowth": ("operating cash flow per share growth", "pct"),
    "netDebtToCashEbitda": ("net debt ex leases ÷ cash EBITDA (operating cash flow + interest)", "x"),
    "cashEbitdaCoverage": ("cash EBITDA ÷ interest", "x"),
    "roeExAoci": ("ROE ex AOCI", "pct"),
    "roe3yExAoci": ("3-year ROE ex AOCI", "pct"),
    "bvpsGrowthExAoci": ("book value per share growth ex AOCI", "pct"),
    "debtToCapitalExAoci": ("debt/capital ex AOCI", "pct"),
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
    band_key: str | None = None  # bands "<metric>@<band_key>" override the shared ones (e.g. life)


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
        band_key=t.get("band_key"),
    )


def template_for(kind: str, captive_finance: bool = False, subtype: str | None = None) -> Template:
    """The template for a business model (a subtype-specific one — e.g. life insurers — wins over
    the kind's general one). Captive-finance groups use the general template with financial health
    unscored (consolidated ratios mix in the finance arm)."""
    key = next(
        (
            k
            for k, t in SCORECARD_TEMPLATES.items()
            if kind in t["kinds"] and subtype in t.get("subtypes", [])
        ),
        None,
    ) or next(
        (k for k, t in SCORECARD_TEMPLATES.items() if kind in t["kinds"] and not t.get("subtypes")),
        "standard",
    )
    t = _tpl(key)
    if captive_finance and "financialHealth" in t.dimensions:
        dims = {k: v for k, v in t.dimensions.items() if k != "financialHealth"}
        unscored = {
            **t.unscored,
            "financialHealth": "not scored — captive-finance arm makes consolidated current ratio, D/E and "
            "coverage not meaningful for the operating business",
        }
        t = Template(t.key, t.label, dims, unscored, t.why, t.valuation, t.display, t.unavailable, t.band_key)
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


def _at(rows: Sequence[dict[str, Any] | None] | None, i: int) -> dict[str, Any]:
    return (rows[i] or {}) if rows and len(rows) > i else {}


def compute_metrics(
    incomes: Sequence[dict[str, Any] | None],
    balances: Sequence[dict[str, Any] | None],
    cashflows: Sequence[dict[str, Any] | None],
    kind: str,
    subtype: str | None = None,
    captive_finance: bool = False,
) -> MetricSet:
    """Every template metric for one company-year, from FMP-named statement dicts given NEWEST
    FIRST ([0] = the year scored, [1] = prior full year, [2] = the one before; missing = {}).
    A metric that is not meaningful is None with a note; one that is missing is just None."""
    inc, pri, pri2 = _at(incomes, 0), _at(incomes, 1), _at(incomes, 2)
    bal, pbal, pbal2 = _at(balances, 0), _at(balances, 1), _at(balances, 2)
    cf, pcf = _at(cashflows, 0), _at(cashflows, 1)
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

    _reit_metrics(m, inc, pri, bal, cf, pcf)
    _insurer_metrics(m, inc, pri, pri2, bal, pbal, pbal2)
    return m


def _dividends(c: dict[str, Any], prior: dict[str, Any] | None = None) -> float | None:
    """Dividends paid (positive): the larger of FMP's common line and total-less-preferred. FMP
    misfiles some lines (O FY2024 common = 0) and no single threshold separates a misfiling from
    genuine preferred (VNO 31%, PEB 90%). One misfiling pattern is detectable: SPG FY2025 files
    $2.8bn of a $3.2bn payout as "preferred", so common collapses to $0.4bn while the TOTAL keeps
    rising — when the estimate falls below half of last year's common dividends while total
    dividends did not fall, the total is used. Dividend-per-share GROWTH is not scored: specials
    and variable policies (PSA, RYN, WY, HST) make it unreliable."""
    common = abs(_f(c.get("commonDividendsPaid")) or 0.0)
    net = abs(_f(c.get("netDividendsPaid")) or 0.0)
    pref = abs(_f(c.get("preferredDividendsPaid")) or 0.0)
    d = max(common, net - pref)
    if prior:
        p_est = _dividends(prior)
        p_net = abs(_f(prior.get("netDividendsPaid")) or 0.0)
        if p_est and d < 0.5 * p_est and net >= p_net > 0:
            d = net  # common filed as "preferred" (SPG FY2025)
    return d if d > 0 else None


def _reit_metrics(
    m: MetricSet,
    inc: dict[str, Any],
    pri: dict[str, Any],
    bal: dict[str, Any],
    cf: dict[str, Any],
    pcf: dict[str, Any],
) -> None:
    """REIT metrics on CASH figures FMP reports reliably — never an FFO reconstruction (FMP Starter
    can't separate impairments, sale gains or discontinued operations; "other income" nets
    interest). Operating cash flow per share is within ~15% of reported FFO/AFFO per share for most
    REITs (O, EQIX, DLR, VICI, LAMR, EXR, MAA, GLPI, PSA, AMT; low for JV-heavy WELL) and can't be
    inflated by gains. Leverage is debt EXCLUDING leases over CASH EBITDA (operating cash flow +
    interest), which impairments and sale gains can't move (CCI FY2024 6.2x vs −20x on operating
    income; BXP FY2025 8.0x vs a gain-flattered 5.3x).
    Guards: operating cash flow per share UP more than REIT_CFO_JUMP is a one-off receipt (VNO
    FY2025 +140%) — every CFO-based metric is n/m (a fall is scored: it is real); a dividend up more
    than REIT_DIVIDEND_JUMP is likely a special (PSA FY2022) — coverage is n/m. Smaller specials
    (RYN FY2025 +45%) are not caught."""
    v = m.values
    cfo, pcfo = (
        _f(cf.get("netCashProvidedByOperatingActivities")),
        _f(pcf.get("netCashProvidedByOperatingActivities")),
    )
    div, pdiv = _dividends(cf, pcf), _dividends(pcf)
    sh, psh = _f(inc.get("weightedAverageShsOutDil")), _f(pri.get("weightedAverageShsOutDil"))
    cps, pcps = _ratio(cfo, sh), _ratio(pcfo, psh)
    # UPWARD jumps only: a one-off receipt flatters every CFO metric; a collapse is real news (SLG
    # FY2025: CFO/share −73%, dividend coverage 0.31x) and must stay visible
    jump = cps is not None and pcps is not None and pcps > 0 and cps / pcps - 1 > REIT_CFO_JUMP
    interest = _f(inc.get("interestExpense"))
    ebitda = cfo + interest if cfo is not None and interest is not None and interest > 0 else None
    cash = _f(bal.get("cashAndCashEquivalents")) or 0.0
    debt = financing_debt(bal) if bal.get("totalDebt") is not None else None
    if jump:
        note = "n/m (operating cash flow per share up more than 50% — one-off receipt suspected)"
        for k in ("cfoPerShareGrowth", "cfoDividendCoverage", "netDebtToCashEbitda", "cashEbitdaCoverage"):
            v[k] = None
            m.notes[k] = note
        return
    v["cfoPerShareGrowth"] = yoy_growth(cps, pcps)
    special = div is not None and pdiv is not None and div > (1 + REIT_DIVIDEND_JUMP) * pdiv
    v["cfoDividendCoverage"] = None if special else _ratio(cfo, div)
    if special:
        m.notes["cfoDividendCoverage"] = "n/m (dividend up more than 50% — special dividend suspected)"
    v["netDebtToCashEbitda"] = _ratio(debt - cash, ebitda) if debt is not None else None
    v["cashEbitdaCoverage"] = _ratio(ebitda, interest) if ebitda is not None and interest else None


def _insurer_metrics(
    m: MetricSet,
    inc: dict[str, Any],
    pri: dict[str, Any],
    pri2: dict[str, Any],
    bal: dict[str, Any],
    pbal: dict[str, Any],
    pbal2: dict[str, Any],
) -> None:
    """Insurer metrics on equity EXCLUDING AOCI (unrealised bond/equity marks), the basis insurers and
    rating agencies use: MET's equity carries an $18bn AOCI loss, which halves reported equity and
    doubles ROE and debt/capital. Multi-year ROE smooths catastrophe years and equity marks."""
    v = m.values

    def eq_x(b: dict[str, Any]) -> float | None:
        """COMMON equity ex AOCI: preferred stock is not the common holders'. Only where FMP fills
        preferredStock (ALL, LNC, EQH, CRBG) — FMP reports 0 for MET and PRU, whose preferred stays
        in (disclosed in the template); ROE uses total net income, not income to common."""
        e = _f(b.get("totalStockholdersEquity"))
        if e is None or e == 0:
            return None
        aoci = _f(b.get("accumulatedOtherComprehensiveIncomeLoss")) or 0.0
        return e - aoci - (_f(b.get("preferredStock")) or 0.0)

    e0, e1, e2 = eq_x(bal), eq_x(pbal), eq_x(pbal2)
    roes = [
        _ratio(_f(i.get("netIncome")), e) if e is not None and e > 0 else None
        for i, e in ((inc, e0), (pri, e1), (pri2, e2))
    ]
    v["roeExAoci"] = roes[0]
    full = [r for r in roes if r is not None]
    v["roe3yExAoci"] = sum(full) / 3 if len(full) == 3 else None  # needs all three years
    sh, psh = _f(inc.get("weightedAverageShsOutDil")), _f(pri.get("weightedAverageShsOutDil"))
    v["bvpsGrowthExAoci"] = yoy_growth(_ratio(e0, sh), _ratio(e1, psh))
    debt = financing_debt(bal) if bal.get("totalDebt") is not None else None
    v["debtToCapitalExAoci"] = (
        _ratio(debt, debt + e0) if debt is not None and e0 is not None and e0 > 0 else None
    )


def _band(value: float | None, metric: str, band_key: str | None = None) -> int | None:
    if value is None:
        return None
    spec = SCORECARD_METRIC_BANDS.get(f"{metric}@{band_key}") or SCORECARD_METRIC_BANDS[metric]
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
            s = _band(val, k, t.band_key)
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
