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
from src.config import DCF_SANITY_BAND, DEFAULT_MAX_PEERS, TARGET_MAX_SINGLE_METHOD_GAP
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


def _price_target(report: dict[str, Any]) -> dict[str, Any]:
    """Blend the valuation methods that are individually credible; show the rest.

    A method enters the blended target only if it is (a) not flagged unreliable by its
    own engine (DCF outside the 0.2x-5x sanity band, P/B without an ROE adjustment) and
    (b) within ±TARGET_MAX_SINGLE_METHOD_GAP of the price. Excluded methods are shown
    with the reason — never silently averaged in (Ford's +2,142% DCF was). If methods
    exist but none qualifies, the valuation is "Indeterminate" rather than a verdict
    built on a number we already know is not credible.
    """
    price = report["stock"]["price"]
    dcf, pv = report["dcf"], report.get("peerValuation")
    lo, hi = DCF_SANITY_BAND
    candidates: list[dict[str, Any]] = []
    if dcf:
        reason = f"outside the {lo:g}×–{hi:g}× price sanity band" if dcf.get("sanityBandBreach") else None
        candidates.append({"name": "DCF", "value": dcf["intrinsicValuePerShare"], "excluded": reason})
    if pv is not None and pv.implied_price is not None:
        reason = None if pv.reliable else "rests on an unadjusted multiple (see peer notes)"
        candidates.append({"name": f"Peer {pv.method_label}", "value": pv.implied_price, "excluded": reason})
    for c in candidates:
        if c["excluded"] is None and price:
            gap = c["value"] / price - 1
            if abs(gap) > TARGET_MAX_SINGLE_METHOD_GAP:
                c["excluded"] = (
                    f"{gap:+.0%} from the price (beyond ±{TARGET_MAX_SINGLE_METHOD_GAP:.0%}) — shown, not blended"
                )

    used = [c for c in candidates if c["excluded"] is None]
    blended = sum(c["value"] for c in used) / len(used) if used else None
    if len(used) == 2:
        method = "blended: average of DCF and peer-multiple"
    elif len(used) == 1:
        method = f"{used[0]['name']} only"
    else:
        method = "unavailable"
    vals = [c["value"] for c in used]
    divergent = len(vals) == 2 and abs(vals[0] - vals[1]) / max(vals) > 0.30
    return {
        "dcf": dcf["intrinsicValuePerShare"] if dcf else None,
        "peerImplied": pv.implied_price if pv is not None else None,
        "blended": blended,
        "method": method,
        "divergent": divergent,
        "candidates": candidates,
        "indeterminate": bool(candidates) and not used,
    }


def _valuation_verdict(target: dict[str, Any], current_price: float | None) -> tuple[str, float | None]:
    """Valuation-only verdict derived strictly from target-vs-price upside —
    deliberately separate from the composite "rating" (which blends quality,
    growth, financial health, and momentum). A high-quality business can
    legitimately be "Rich" on valuation and vice versa; collapsing both into
    one label is how you end up with a nonsensical "Buy, -36% target".
    """
    if target.get("indeterminate"):
        return "Indeterminate (methods not credible or disagree)", None
    blended_target = target.get("blended")
    if blended_target is None or not current_price:
        return "n/a", None
    upside = (blended_target - current_price) / current_price
    if upside >= 0.15:
        return "Attractive (undervalued)", upside
    if upside >= -0.10:
        return "Fair value", upside
    return "Rich (overvalued)", upside


def _reconcile(rating: str, valuation_verdict: str) -> str:
    """Explicit rating-vs-valuation reconciliation (three-way, not a sign test)."""
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
        out.append("| Symbol | Market cap | P/E | P/B | ROE |")
        out.append("|---|---|---|---|---|")
        for _, r in pv.peers.iterrows():
            pe = f"{r['pe']:.1f}" if pd.notna(r.get("pe")) else "n/a"
            pb = f"{r['pb']:.2f}" if pd.notna(r.get("pb")) else "n/a"
            roe = f"{r['roe']:.1%}" if pd.notna(r.get("roe")) else "n/a"
            out.append(f"| {r['symbol']} | {money(r.get('marketCap'))} | {pe} | {pb} | {roe} |")
    return out


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
    if stock.get("businessModel", "standard") in FINANCIAL_KINDS:
        # Current ratio and D/E are meaningless for deposit-/float-funded balance sheets.
        eta = leverage.get("equityToAssets")
        if eta is not None and eta < 0.08:
            risks.append(f"Thin capital base: equity is {eta:.1%} of total assets.")
    else:
        if leverage["debtToEquityRatio"] is not None and leverage["debtToEquityRatio"] > 1.0:
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
    if dcf and not dcf.get("sanityBandBreach") and dcf["upsidePct"] is not None and dcf["upsidePct"] < -0.25:
        risks.append(
            f"Trades {abs(dcf['upsidePct']):.0%} above DCF intrinsic value — "
            "meaningful valuation risk if growth/margin assumptions disappoint."
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


def _executive_summary(report: dict[str, Any], target: dict[str, Any], valuation_verdict: str) -> list[str]:
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

    if target["blended"] is not None and stock["price"]:
        upside = (target["blended"] - stock["price"]) / stock["price"]
        direction = "upside" if upside >= 0 else "downside"
        bullets.append(
            f"Price target ${target['blended']:.2f} ({target['method']}) implies {abs(upside):.1%} "
            f"{direction} from the current ${stock['price']:.2f}."
        )

    if target["divergent"]:
        bullets.append(
            f"⚠️ DCF (${target['dcf']:.2f}) and peer-multiple (${target['peerImplied']:.2f}) valuations "
            "diverge significantly — the market may be pricing in growth/quality the DCF's assumptions "
            "don't fully capture, or the stock is richly valued relative to fundamentals. See Valuation section."
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
            f"  |  Target: ${target['blended']:.2f} ({target['method']})"
            if target["blended"] is not None
            else ""
        )
    )
    lines.append("")

    lines.append("## Executive Summary")
    lines.extend(f"- {b}" for b in _executive_summary(report, target, valuation_verdict))
    lines.append("")

    lines.append("## Company Overview")
    lines.append(f"- Sector: {stock['sector']} / {stock['industry']}")
    if stock["marketCap"]:
        lines.append(f"- Market cap: {money(stock['marketCap'], stock.get('quoteCurrency'))}")
    if stock["currencyMismatch"]:
        lines.append(
            f"- ⚠️ Quoted in {stock['quoteCurrency']}, reports financials in {stock['reportingCurrency']}"
        )
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
    for c in target["candidates"]:
        status = f"excluded — {c['excluded']}" if c["excluded"] else "used"
        lines.append(f"- {c['name']}: ${c['value']:.2f} ({status})")
    if target["indeterminate"]:
        lines.append(
            "**Valuation: Indeterminate** — every available method is either flagged as not credible for this "
            "company or sits beyond ±50% of the price, so no price target is published. Read the individual "
            "methods above as ranges to investigate, not as a verdict."
        )
    elif target["blended"] is not None:
        upside = (target["blended"] - stock["price"]) / stock["price"]
        lines.append(
            f"**Valuation: {valuation_verdict}** — blended price target **${target['blended']:.2f}** "
            f"({target['method']}), implying {upside:+.1%} vs the current price of ${stock['price']:.2f}."
        )
        if target["divergent"]:
            lines.append(
                "- DCF and peer-multiple methodologies disagree meaningfully here — the market may be "
                "pricing in growth/quality the DCF's conservative assumptions don't capture, or the stock "
                "is priced for perfection relative to fundamentals. Worth forming your own view on which "
                "better fits this business before acting on either number alone."
            )
        lines.append(
            f"\n**Bottom line:** {stock['companyName']} is a {score['rating'].lower()}-quality business "
            f"({score['compositeScore']}/100) trading at a valuation assessed as {valuation_verdict.lower()}. "
            + _reconcile(score["rating"], valuation_verdict)
        )
    else:
        lines.append("**Valuation: n/a** — no reliable price target available (see Key Risks for why).")

    return "\n".join(lines)
