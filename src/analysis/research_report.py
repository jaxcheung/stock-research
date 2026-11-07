# src/analysis/research_report.py
#
# Ties fundamentals, peer comps, DCF valuation, macro backdrop, and the
# investment scorecard into one equity-research-style report: executive
# summary, financial performance trend, valuation triangulation (DCF +
# peer multiples), identified risks, and a final rating + price target.
#
# Valuation inputs: run_dcf (refuses business models it doesn't fit) and
# relative_valuation.peer_valuation (business-model-appropriate multiple, same-
# model peers). Both are computed once here and passed to the scorecard.

from typing import Any

import pandas as pd

from src.analysis.business_model import FINANCIAL_KINDS
from src.analysis.dcf import run_dcf
from src.analysis.investment_score import compute_investment_score
from src.analysis.macro_analysis import get_macro_backdrop
from src.analysis.relative_valuation import PeerValuation, peer_valuation
from src.analysis.stock_analysis import analyze_stock
from src.analysis.valuation_summary import ValuationSummary, summarize_valuation
from src.config import DEFAULT_MAX_PEERS
from src.data.fmp_client import FMPClient, FMPError
from src.data.fred_client import FREDError
from src.reporting.formatting import money
from src.utils.logging import get_logger

logger = get_logger("research_report")


def generate_research_report(
    symbol: str, client: FMPClient | None = None, max_peers: int = DEFAULT_MAX_PEERS
) -> dict[str, Any]:
    """Assemble all inputs for a full equity research report."""
    client = client or FMPClient()

    stock = analyze_stock(symbol, client=client)

    dcf_unavailable_reason: str | None = None
    try:
        dcf_result = run_dcf(symbol, client=client)
    except (ValueError, FMPError) as exc:
        logger.warning("DCF unavailable for %s: %s", symbol, exc)
        dcf_result = None
        dcf_unavailable_reason = str(exc)

    peer_val: PeerValuation | None = None
    peer_error: str | None = None
    try:
        peer_val = peer_valuation(symbol, client=client, max_peers=max_peers)
    except (ValueError, FMPError) as exc:
        logger.warning("Peer valuation unavailable for %s: %s", symbol, exc)
        peer_error = str(exc)

    try:
        macro = get_macro_backdrop()
    except FREDError as exc:
        logger.warning("Macro backdrop unavailable: %s", exc)
        macro = None

    score_result = compute_investment_score(
        symbol,
        client=client,
        max_peers=max_peers,
        inputs={"stock": stock, "dcf": dcf_result, "peers": peer_val},
    )

    return {
        "symbol": stock["symbol"],
        "stock": stock,
        "dcf": dcf_result,
        "dcfUnavailableReason": dcf_unavailable_reason,
        "peerValuation": peer_val,
        "peerError": peer_error,
        "macro": macro,
        "score": score_result,
    }


def _price_target(report: dict[str, Any]) -> ValuationSummary:
    """Which valuation methods count and the resulting target / verdict — delegated to
    valuation_summary so the scorecard's valuation signal uses the identical rule."""
    return summarize_valuation(report["stock"]["price"], report["dcf"], report.get("peerValuation"))


def _valuation_verdict(target: ValuationSummary, current_price: float | None) -> tuple[str, float | None]:
    """Valuation-only verdict — deliberately separate from the composite "rating" (which
    blends quality, growth, financial health, and momentum). A high-quality business can
    legitimately be "Rich" on valuation and vice versa; collapsing both into one label is
    how you end up with a nonsensical "Buy, -36% target"."""
    return target.verdict, target.upside


def _reconcile(rating: str, valuation_verdict: str) -> str:
    """Explicit rating-vs-valuation reconciliation (three-way, not a sign test). A verdict
    resting on one method says so — "quality and price agree" needs a confirmed price view."""
    text = _reconcile_core(rating, valuation_verdict)
    if "(single method)" in valuation_verdict or "low precision" in valuation_verdict:
        qualifier = (
            "one unconfirmed method" if "(single method)" in valuation_verdict else "imprecise estimates"
        )
        text = f"On {qualifier}: " + text[0].lower() + text[1:]
    return text


def _reconcile_core(rating: str, valuation_verdict: str) -> str:
    good_business = rating in ("Strong Buy", "Buy")
    weak_business = rating in ("Sell", "Strong Sell")
    cheap = valuation_verdict.startswith("Attractive")
    rich = valuation_verdict.startswith("Rich")
    if good_business and cheap:
        return "Quality and price agree — a strong business at an attractive price."
    if weak_business and rich:
        return "Quality and price agree — a weak business that also looks expensive."
    if good_business and rich:
        return (
            "A strong business, but the price already discounts it — quality alone is not a reason "
            "to buy at this level."
        )
    if weak_business and cheap:
        return "Looks cheap, but the business is weak — check for a value trap before acting."
    if cheap:
        return (
            "An average-quality business that looks cheap — the valuation case depends on the discount "
            "being a mispricing rather than a fair reflection of middling fundamentals."
        )
    if rich:
        return (
            "An average-quality business that looks expensive — the price assumes improvement the "
            "fundamentals don't yet show."
        )
    return (
        "Neither quality nor valuation gives a decisive signal — the case rests on your own view "
        "of the assumptions; don't rely on the composite rating alone as a buy/sell signal."
    )


def _peer_section(report: dict[str, Any]) -> list[str]:
    pv: PeerValuation | None = report.get("peerValuation")
    if pv is None:
        return [f"- Unavailable: {report.get('peerError') or 'peer valuation failed'}"]
    out = [
        f"*Method: {pv.method_label} — chosen for a {pv.kind.replace('_', ' ')} business. "
        f"Peer set: {pv.scope or 'none'} (same business model, US listings, operating equities only).*"
    ]
    if pv.industries_screened and pv.scope in ("Industry peers", "Industry + related peers"):
        out.append("- FMP industries screened: " + "; ".join(pv.industries_screened))
    f = lambda v: f"{v:.2f}×" if v is not None else "n/a"  # noqa: E731
    if pv.n:
        out.append(
            f"- Peer aggregate (Σ cap ÷ Σ denominator): {f(pv.peer_aggregate)} · median {f(pv.peer_median)} · "
            f"n = {pv.n} · {pv.symbol}: {f(pv.subject_multiple)}"
            + (f" · warranted (ROE-adjusted): {f(pv.target_multiple)}" if pv.roe_adjustment else "")
        )
    if pv.implied_price is not None:
        out.append(f"- Peer-implied value: **${pv.implied_price:.2f}**")
    if pv.unavailable_reason:
        out.append(f"- No peer-implied value: {pv.unavailable_reason}")
    out.extend(f"- {n}" for n in pv.notes)
    if pv.excluded_model:
        out.append("- Excluded (different business model): " + ", ".join(pv.excluded_model))
    if pv.excluded_fx:
        out.append("- Excluded (quote/reporting currency mismatch): " + ", ".join(pv.excluded_fx))
    if not pv.peers.empty:
        out.append(
            "| Symbol | FMP industry | Market cap | P/E | P/B | ROE (TTM earnings ÷ book) | Book source |"
        )
        out.append("|---|---|---|---|---|---|---|")
        for _, r in pv.peers.iterrows():
            pe = f"{r['pe']:.1f}" if pd.notna(r.get("pe")) else "n/a"
            pb = f"{r['pb']:.2f}" if pd.notna(r.get("pb")) else "n/a"
            e, b = r.get("earnings"), r.get("book")
            roe = f"{e / b:.1%}" if pd.notna(e) and pd.notna(b) and b else "n/a"
            out.append(
                f"| {r['symbol']} | {r.get('industry') or ''} | {money(r.get('marketCap'))} | {pe} | {pb} | {roe} "
                f"| {r.get('bookSource') or 'TTM'} |"
            )
    return out


def _fmt_range(r: tuple[float, float]) -> str:
    lo_v, hi_v = r
    return f"${lo_v:.2f}" if abs(hi_v - lo_v) < 0.005 else f"${lo_v:.2f}–${hi_v:.2f}"


def _identify_risks(report: dict[str, Any]) -> list[str]:
    risks = []
    stock = report["stock"]
    leverage = stock["leverage"]
    growth = stock["growth"]
    profitability = stock["profitability"]

    if stock["currencyMismatch"]:
        risks.append(
            f"Reports financials in {stock['reportingCurrency']} while quoted in {stock['quoteCurrency']} "
            "— valuation multiples and DCF are unavailable without FX conversion."
        )

    equity = leverage.get("totalEquity")
    if equity is not None and equity <= 0:
        risks.append(
            f"Negative shareholders' equity ({money(equity, stock.get('reportingCurrency'))}) — typically "
            "buybacks/dividends funded with debt; debt/equity and ROE are not meaningful, and the balance "
            "sheet has no equity cushion."
        )
    kind = stock.get("businessModel", "standard")
    if kind in FINANCIAL_KINDS:
        # Current ratio and D/E are meaningless for deposit-/float-funded balance sheets;
        # equity/assets is meaningless for insurers (separate-account assets).
        eta = leverage.get("equityToAssets")
        if kind != "insurer" and eta is not None and eta < 0.08:
            risks.append(f"Thin capital base: equity is {eta:.1%} of total assets.")
    else:
        if stock.get("captiveFinance"):
            risks.append(
                "Consolidated statements include a captive-finance arm: reported debt, debt/equity and "
                "interest coverage mix the lending business with the operating company — judge leverage "
                "on the industrial segment (not separated in FMP data)."
            )
        elif leverage["debtToEquityRatio"] is not None and leverage["debtToEquityRatio"] > 1.0:
            risks.append(f"Elevated leverage: debt/equity of {leverage['debtToEquityRatio']:.2f}.")
        if leverage["currentRatio"] is not None and leverage["currentRatio"] < 1.0:
            risks.append(f"Liquidity risk: current ratio of {leverage['currentRatio']:.2f} is below 1.0x.")

    if growth["revenueGrowthYoY"] is not None and growth["revenueGrowthYoY"] < 0:
        risks.append(f"Revenue declined {abs(growth['revenueGrowthYoY']):.1%} YoY.")
    if growth["netIncomeGrowthYoY"] is not None and growth["netIncomeGrowthYoY"] < 0:
        risks.append(f"Net income declined {abs(growth['netIncomeGrowthYoY']):.1%} YoY.")

    if profitability["netProfitMargin"] is not None and profitability["netProfitMargin"] < 0:
        risks.append("Currently unprofitable (negative net margin).")

    dcf = report["dcf"]
    summary = _price_target(report)
    dcf_credible = any(m.name == "DCF" and m.eligible for m in summary.methods)
    if dcf and dcf_credible and dcf["upsidePct"] is not None and dcf["upsidePct"] < -0.25:
        premium = dcf["currentPrice"] / dcf["intrinsicValuePerShare"] - 1
        risks.append(
            f"Price is {premium:.0%} above DCF intrinsic value (intrinsic value {abs(dcf['upsidePct']):.0%} "
            "below the price) — meaningful valuation risk if growth/margin assumptions disappoint."
        )
    rich_far = [
        m
        for m in summary.methods
        if m.eligible and m.upside is not None and m.upside < -0.5 and m.name != "DCF"
    ]
    if summary.precision in ("low", "indicative") and rich_far and not summary.verdict.endswith("disagree"):
        names = " and ".join(f"{m.name} (${m.value:.2f}, {m.upside:+.0%})" for m in rich_far)
        confirmed = (
            "confirmed by both methods" if summary.precision == "low" else "single method, unconfirmed"
        )
        risks.append(
            f"Valuation risk: {names} values the stock more than 50% below the price of ${stock['price']:.2f} "
            f"({confirmed}) — the price embeds expectations well beyond what this method supports."
        )

    dq = stock.get("dataQuality")
    # The balance-sheet stub already has its own line (dataNotes); don't count it twice.
    dq_warn = [
        i
        for i in (dq.warnings if dq else [])
        # only the balance-sheet stub already has its own line (dataNotes)
        if not (i.check == "stub row" and i.message.startswith("balance sheet") and stock.get("dataNotes"))
    ]
    if dq is not None and dq_warn:
        risks.append(
            f"Data quality: {len(dq_warn)} warning(s) in the source statements "
            f"({', '.join(dict.fromkeys(i.check for i in dq_warn))}) — see the Data Quality section "
            "before relying on the affected figures."
        )

    if report["macro"] and report["macro"]["yieldCurveInverted"]:
        risks.append(
            "Yield curve is inverted — a historical leading recession indicator that could "
            "pressure cyclical earnings and risk appetite broadly."
        )

    if not risks:
        risks.append(
            "No material red flags identified in the metrics reviewed; standard market and execution risks still apply."
        )

    return risks


def _executive_summary(report: dict[str, Any], target: ValuationSummary, valuation_verdict: str) -> list[str]:
    stock, score = report["stock"], report["score"]
    bullets = [f"**Fundamental quality: {score['rating']}** — composite score {score['compositeScore']}/100."]
    bullets.append(
        f"**Valuation: {valuation_verdict}** (see price target below) — a separate question from quality."
    )

    val = score.get("valuation")
    if val and val["score"] is not None:
        bullets.append(
            f"Valuation signal: {val['score']}/100 (not part of the quality rating) — {val['rationale']}"
        )

    dims_by_score = sorted(
        ((name, d) for name, d in score["dimensions"].items() if d["score"] is not None),
        key=lambda item: item[1]["score"],
        reverse=True,
    )
    if dims_by_score:
        best_name, best = dims_by_score[0]
        worst_name, worst = dims_by_score[-1]
        bullets.append(f"Strongest factor: **{best_name}** ({best['score']}/100) — {best['rationale']}")
        bullets.append(f"Weakest factor: **{worst_name}** ({worst['score']}/100) — {worst['rationale']}")

    if target.target is not None and target.upside is not None and stock["price"]:
        direction = "upside" if target.upside >= 0 else "downside"
        bullets.append(
            f"Price target ${target.target:.2f} ({target.method_text}) implies {abs(target.upside):.1%} "
            f"{direction} from the current ${stock['price']:.2f}."
        )
    elif target.target_range is not None:
        lo_v, hi_v = target.target_range
        bullets.append(
            f"No point target: {target.method_text} ({_fmt_range((lo_v, hi_v))} vs ${stock['price']:.2f})."
        )

    if target.divergent and target.divergence_note:
        bullets.append(
            f"⚠️ Valuation methods diverge ({target.divergence_note}) — the market may be pricing in "
            "growth/quality one method doesn't capture, or the stock is richly valued relative to "
            "fundamentals. See Valuation section."
        )

    return bullets


def render_research_report(
    symbol: str, client: FMPClient | None = None, max_peers: int = DEFAULT_MAX_PEERS
) -> str:
    """Render generate_research_report() output as a full markdown equity research report."""
    report = generate_research_report(symbol, client=client, max_peers=max_peers)
    stock, dcf, macro, score = (
        report["stock"],
        report["dcf"],
        report["macro"],
        report["score"],
    )
    target = _price_target(report)
    valuation_verdict, _ = _valuation_verdict(target, stock["price"])

    lines: list[str] = []
    lines.append(f"# Equity Research Report: {stock['companyName']} ({stock['symbol']})")
    lines.append(
        f"**Fundamental Quality: {score['rating']}** ({score['compositeScore']}/100)  |  "
        f"**Valuation: {valuation_verdict}**  |  Price: ${stock['price']:.2f}"
        + (
            f"  |  Target: ${target.target:.2f} ({target.method_text})"
            if target.target is not None
            else (
                f"  |  Indicative only, not a target: {_fmt_range(target.target_range)} ({target.precision} precision)"
                if target.target_range is not None
                else ""
            )
        )
    )
    lines.append("")

    lines.append("## Executive Summary")
    lines.extend(f"- {b}" for b in _executive_summary(report, target, valuation_verdict))
    lines.append("")

    lines.append("## Company Overview")
    lines.append(f"- Sector: {stock['sector']} / {stock['industry']}")
    for note in stock.get("dataNotes", []):
        lines.append(f"- ⚠️ Data quality: {note}")
    lines.append(
        f"- Business model: {stock.get('businessModel', 'standard').replace('_', ' ')}"
        + (" with captive finance" if stock.get("captiveFinance") else "")
        + f" ({stock.get('businessModelReason', '')})"
    )
    if stock["marketCap"]:
        lines.append(f"- Market cap: {money(stock['marketCap'], stock.get('quoteCurrency'))}")
    if stock["currencyMismatch"]:
        lines.append(
            f"- ⚠️ Quoted in {stock['quoteCurrency']}, reports financials in {stock['reportingCurrency']}"
        )
    lines.append("")

    dq = stock.get("dataQuality")
    if dq is not None:
        lines.append("## Data Quality")
        lines.append(
            f"*{dq.summary()} Deterministic checks on the source statements; reported, never corrected.*"
        )
        lines.extend(f"- {line}" for line in dq.lines())
        lines.append("")

    lines.append("## Macro Backdrop")
    if macro:
        gdp = macro["realGdpGrowth"]
        cpi = macro["cpiYoY"]
        lines.append(f"- Real GDP growth (SAAR): {gdp['value']:.2f}%" if gdp else "- Real GDP growth: n/a")
        lines.append(f"- CPI inflation (YoY): {cpi['value']:.2%}" if cpi else "- CPI inflation: n/a")
        lines.append(
            f"- 10Y Treasury yield: {macro['treasury10Y']['value']:.2f}%"
            if macro["treasury10Y"]
            else "- 10Y Treasury yield: n/a"
        )
        if macro["yieldCurveInverted"]:
            lines.append("- ⚠️ Yield curve inverted — historical recession signal")
    else:
        lines.append("- Unavailable (FRED not configured)")
    lines.append("")

    lines.append("## Financial Performance")
    history = sorted(stock["revenueHistory"], key=lambda r: r["date"])
    if history:
        ccy = stock.get("reportingCurrency") or stock.get("quoteCurrency")
        lines.append(f"*Amounts in reporting currency ({ccy or 'n/a'})*")
        lines.append("| Year | Revenue | Net Income | Net Margin |")
        lines.append("|---|---|---|---|")
        for row in history:
            margin = (
                row["netIncome"] / row["revenue"]
                if row.get("revenue") and row.get("netIncome") is not None
                else None
            )
            margin_str = f"{margin:.1%}" if margin is not None else "n/a"
            lines.append(
                f"| {row['date']} | {money(row.get('revenue'), ccy)} | {money(row.get('netIncome'), ccy)} | {margin_str} |"
            )
    else:
        lines.append("No historical data available.")
    lines.append("")

    lines.append("## Valuation")
    lines.append("### DCF")
    if dcf:
        lines.append(
            f"- WACC {dcf['wacc']:.2%}, FCF growth {dcf['assumptions']['fcfGrowthRate']:.2%}"
            + (" (historical CAGR, clamped)" if dcf["assumptions"].get("fcfGrowthRateClamped") else "")
            + f", terminal growth {dcf['assumptions']['terminalGrowthRate']:.2%}, "
            f"risk-free rate {dcf['assumptions']['riskFreeRate']:.2%} ({dcf['assumptions']['riskFreeRateSource']})"
        )
        lines.append(
            f"- Intrinsic value/share: **${dcf['intrinsicValuePerShare']:.2f}** vs price ${dcf['currentPrice']:.2f} "
            f"({dcf['upsidePct']:+.1%})"
        )
        if dcf.get("terminalValueShare") is not None:
            lines.append(
                f"- Terminal value is {dcf['terminalValueShare']:.0%} of enterprise value"
                + (
                    " — ⚠️ the result is mostly the terminal-growth assumption"
                    if dcf.get("terminalValueShareFlag")
                    else ""
                )
            )
        if dcf["assumptions"].get("costOfDebtFloored"):
            lines.append(
                f"- Cost of debt {dcf['assumptions']['costOfDebt']:.2%} = risk-free + spread: the derived "
                f"{dcf['assumptions']['derivedCostOfDebt']:.2%} (interest ÷ debt) reflects legacy coupons, "
                "not the marginal cost of borrowing"
            )
        if dcf.get("sanityBandBreach"):
            lines.append("- ⚠️ Outside the 0.2×–5× price sanity band — shown, not used in the price target")
    else:
        reason = report.get("dcfUnavailableReason")
        lines.append(f"- Unavailable: {reason}" if reason else "- Unavailable")
    lines.append("")

    lines.append("### Peer Comparison")
    lines.extend(_peer_section(report))
    lines.append("")

    lines.append("## Investment Scorecard")
    lines.append(
        "*Fundamental-quality composite (weights in `src/config.py:SCORE_WEIGHTS`). "
        "The valuation signal is shown separately and is not part of the rating.*"
    )
    lines.append("| Dimension | Score | Rationale |")
    lines.append("|---|---|---|")
    for name, d in score["dimensions"].items():
        score_str = f"{d['score']}/100" if d["score"] is not None else "n/a"
        lines.append(f"| {name} | {score_str} | {d['rationale']} |")
    val = score.get("valuation")
    if val:
        val_str = f"{val['score']}/100" if val["score"] is not None else "n/a"
        lines.append(f"| *valuation signal (excluded)* | {val_str} | {val['rationale']} |")
    lines.append("")

    lines.append("## Key Risks")
    lines.extend(f"- {r}" for r in _identify_risks(report))
    lines.append("")

    lines.append("## Rating & Price Target")
    lines.append(
        f"**Fundamental quality: {score['rating']}** ({score['compositeScore']}/100) — reflects quality, "
        "growth, financial health, and price momentum. This is a view on the business, not the entry price."
    )
    for m in target.methods:
        reason = f"not used — {m.reason}" if m.reason else "not used"
        status = "used" + (f" ({m.reason})" if m.reason else "") if m in target.used else reason
        up = f", {m.upside:+.0%}" if m.upside is not None else ""
        lines.append(f"- {m.name}: ${m.value:.2f}{up} ({status})")
    if target.target is not None and target.upside is not None:
        lines.append(
            f"**Valuation: {valuation_verdict}** — price target **${target.target:.2f}** "
            f"({target.method_text}), implying {target.upside:+.1%} vs the current price of ${stock['price']:.2f}."
        )
    elif target.precision == "low" and target.target_range is not None:
        lo_v, hi_v = target.target_range
        lines.append(
            f"**Valuation: {valuation_verdict}** — {target.method_text}: {_fmt_range((lo_v, hi_v))} vs "
            f"${stock['price']:.2f}. The direction is consistent across methods; the magnitude is not "
            "precise enough for a point target."
        )
    elif target.precision == "indicative" and target.target_range is not None:
        lines.append(
            f"**Valuation: {valuation_verdict}** — {target.method_text}. One method on its own, this far "
            "from the price, is a lead to investigate (check that method's inputs above), not a verdict; "
            "it is not scored."
        )
    elif target.indeterminate:
        lines.append(
            f"**Valuation: {valuation_verdict}** — "
            + (
                "the credible methods point in different directions, so no verdict or target is published."
                if target.verdict.endswith("disagree")
                else "every available method is flagged as not credible for this company (reasons above)."
            )
        )
    else:
        lines.append("**Valuation: n/a** — no valuation method is available (see the Valuation section).")
    if target.divergent and target.divergence_note:
        lines.append(
            f"- Methods diverge ({target.divergence_note}) — form your own view on which better fits "
            "this business before acting on either number alone."
        )
    if target.precision in ("normal", "single", "low"):
        lines.append(
            f"\n**Bottom line:** {stock['companyName']} is a {score['rating'].lower()}-quality business "
            f"({score['compositeScore']}/100) trading at a valuation assessed as {valuation_verdict.lower()}. "
            + _reconcile(score["rating"], valuation_verdict)
        )

    return "\n".join(lines)
