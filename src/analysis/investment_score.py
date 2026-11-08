# src/analysis/investment_score.py
#
# CFA-style fundamental-quality scorecard: blends profitability quality,
# growth, financial health, and price momentum into one 0-100 score and a
# Buy/Hold/Sell rating. Valuation (DCF + peer multiples) is scored too, but
# reported as a separate signal and kept OUT of the composite — the rating is
# a view on the business, not the entry price (CLAUDE.md Rules). Each dimension is scored
# independently with an explicit banding methodology (disclosed in the
# rationale) so the rating is auditable, not a black box. A dimension with
# insufficient data is dropped and the remaining weights are renormalized
# rather than guessing a neutral score.

from typing import Any

import pandas as pd

from src.analysis.dcf import run_dcf
from src.analysis.relative_valuation import PeerValuation, peer_valuation
from src.analysis.stock_analysis import analyze_stock
from src.analysis.templates import MetricSet, score_dimensions, template_for
from src.analysis.valuation_summary import summarize_valuation
from src.config import (
    DEFAULT_MAX_PEERS,
    SCORE_RATING_BANDS,
    SCORE_WEIGHTS,
    VALUATION_SCORE_CAP,
)
from src.data.fmp_client import FMPClient, FMPError
from src.utils.logging import get_logger

logger = get_logger("investment_score")

# (threshold, score) bands, evaluated highest-threshold-first: value >= threshold wins.
_UPSIDE_BANDS = [(0.25, 100), (0.15, 85), (0.05, 70), (0.0, 55), (-0.10, 35), (-0.25, 15), (float("-inf"), 0)]


def _band_score_higher_better(value: float | None, bands: list[tuple[float, int]]) -> int | None:
    if value is None:
        return None
    for threshold, score in bands:
        if value >= threshold:
            return score
    return bands[-1][1]


def _band_score_lower_better(value: float | None, bands: list[tuple[float, int]]) -> int | None:
    if value is None:
        return None
    for threshold, score in bands:
        if value <= threshold:
            return score
    return bands[-1][1]


def _score_valuation(
    stock: dict[str, Any], dcf_result: dict[str, Any] | None, peer_val: PeerValuation | None
) -> dict[str, Any]:
    """Valuation signal from exactly the methods behind the report's valuation verdict
    (valuation_summary) — a method the price target excludes is never scored."""
    summary = summarize_valuation(stock.get("price"), dcf_result, peer_val)
    detail: dict[str, Any] = {"verdict": summary.verdict, "precision": summary.precision}
    sub_scores = []
    notes = []
    for m in summary.methods:
        if m.upside is None:
            continue
        detail[f"{m.name} upside"] = m.upside
        direction = "upside" if m.upside >= 0 else "downside"
        text = f"{m.name} implies {abs(m.upside):.0%} {direction}"
        if m in summary.used:
            sub_scores.append(_band_score_higher_better(m.upside, _UPSIDE_BANDS))
            notes.append(text)
        else:
            notes.append(f"{text} — not scored ({m.reason})")
    if peer_val is not None and peer_val.implied_price is None and peer_val.unavailable_reason:
        notes.append(f"No peer valuation: {peer_val.unavailable_reason}")

    valid_scores = [s for s in sub_scores if s is not None]
    score = round(sum(valid_scores) / len(valid_scores)) if valid_scores else None
    if summary.scoring == "none":
        score = None
        notes.append(f"not scored: {summary.verdict}")
    elif summary.scoring == "capped" and score is not None:
        lo, hi = VALUATION_SCORE_CAP
        basis = "single method" if summary.precision == "single" else "low precision"
        notes.append(f"{basis} — score limited to {lo}–{hi}")
        score = min(max(score, lo), hi)
    return {"score": score, "rationale": "; ".join(notes) if notes else "Insufficient data", "detail": detail}


def _score_momentum(stock: dict[str, Any]) -> dict[str, Any]:
    price = stock["price"]
    tech = stock["technical"]
    avg50, avg200 = tech["priceAvg50"], tech["priceAvg200"]

    if price is None or avg50 is None or avg200 is None:
        return {"score": None, "rationale": "Insufficient data", "detail": tech}

    above_50 = price > avg50
    above_200 = price > avg200
    if above_50 and above_200:
        score, trend = 100, "uptrend (above both 50D and 200D averages)"
    elif above_200 and not above_50:
        score, trend = 60, "pullback within a longer-term uptrend"
    elif above_50 and not above_200:
        score, trend = 40, "short-term bounce within a longer-term downtrend"
    else:
        score, trend = 0, "downtrend (below both 50D and 200D averages)"

    return {"score": score, "rationale": f"Price ${price:.2f} — {trend}", "detail": tech}


def _rating_from_score(score: float) -> str:
    for threshold, rating in SCORE_RATING_BANDS:
        if score >= threshold:
            return rating
    return SCORE_RATING_BANDS[-1][1]


def score_business(stock: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], float | None, str | None]:
    """The fundamental-quality rule, shared by the live scorecard and the backtester. Which metrics
    are scored is decided by the company's industry TEMPLATE (templates.template_for); business
    dimensions only (valuation is reported alongside, never blended in), each banded, missing
    dimensions dropped and the remaining weights renormalised. Returns (dimensions, composite
    0-100 or None when nothing is scoreable, rating or None)."""
    tpl = template_for(stock.get("businessModel", "standard"), bool(stock.get("captiveFinance")))
    metrics = stock.get("metrics")
    if not isinstance(metrics, MetricSet):
        raise ValueError("stock['metrics'] missing: build it with templates.compute_metrics")
    business = score_dimensions(tpl, metrics)
    dimensions = {
        "quality": business.get("quality", {"score": None, "rationale": "Insufficient data", "detail": {}}),
        "growth": business.get("growth", {"score": None, "rationale": "Insufficient data", "detail": {}}),
        "financialHealth": business.get(
            "financialHealth", {"score": None, "rationale": "Insufficient data", "detail": {}}
        ),
        "momentum": _score_momentum(stock),
    }
    scored = {name: d["score"] for name, d in dimensions.items() if d["score"] is not None}
    if not scored:
        return dimensions, None, None
    total_weight = sum(SCORE_WEIGHTS[name] for name in scored)
    composite = sum(scored[name] * SCORE_WEIGHTS[name] for name in scored) / total_weight
    return dimensions, composite, _rating_from_score(composite)


def compute_investment_score(
    symbol: str,
    client: FMPClient | None = None,
    max_peers: int = DEFAULT_MAX_PEERS,
    inputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Composite CFA-style investment score and Buy/Hold/Sell rating for one symbol.

    `inputs` ({"stock", "dcf", "peers"}) lets the research report pass results it already
    computed instead of re-running them; a None "dcf"/"peers" means "unavailable".
    """
    client = client or FMPClient()
    if inputs is not None:
        stock, dcf_result, peer_val = inputs["stock"], inputs.get("dcf"), inputs.get("peers")
    else:
        stock = analyze_stock(symbol, client=client)
        try:
            dcf_result = run_dcf(symbol, client=client)
        except (ValueError, FMPError) as exc:
            logger.warning("DCF unavailable for %s: %s", symbol, exc)
            dcf_result = None
        try:
            peer_val = peer_valuation(symbol, client=client, max_peers=max_peers)
        except (ValueError, FMPError) as exc:
            logger.warning("Peer valuation unavailable for %s: %s", symbol, exc)
            peer_val = None

    dimensions, composite_score, rating = score_business(stock)
    if composite_score is None:
        raise ValueError(f"No scoreable dimensions available for {symbol}")

    return {
        "symbol": stock["symbol"],
        "companyName": stock["companyName"],
        "currentPrice": stock["price"],
        "compositeScore": round(composite_score, 1),
        "rating": rating,
        "dimensions": dimensions,
        "valuation": _score_valuation(stock, dcf_result, peer_val),
        "dcfIntrinsicValue": dcf_result["intrinsicValuePerShare"] if dcf_result else None,
        "dcfUpsidePct": dcf_result["upsidePct"] if dcf_result else None,
    }


def rank_opportunities(
    symbols: list[str], client: FMPClient | None = None, max_peers: int = DEFAULT_MAX_PEERS
) -> pd.DataFrame:
    """Score multiple symbols and rank them best-to-worst — a simple screener."""
    client = client or FMPClient()
    rows = []
    for symbol in symbols:
        try:
            result = compute_investment_score(symbol, client=client, max_peers=max_peers)
        except (ValueError, KeyError, FMPError) as exc:
            logger.warning("Skipping %s in ranking: %s", symbol, exc)
            continue
        rows.append(
            {
                "symbol": result["symbol"],
                "companyName": result["companyName"],
                "compositeScore": result["compositeScore"],
                "rating": result["rating"],
                "valuationScore": result["valuation"]["score"],
                "valuationVerdict": result["valuation"]["detail"].get("verdict"),
                "valuationPrecision": result["valuation"]["detail"].get("precision"),
                "currentPrice": result["currentPrice"],
                "dcfIntrinsicValue": result["dcfIntrinsicValue"],
                "dcfUpsidePct": result["dcfUpsidePct"],
            }
        )
    columns = [
        "symbol",
        "companyName",
        "compositeScore",
        "rating",
        "valuationScore",
        "valuationVerdict",
        "valuationPrecision",
        "currentPrice",
        "dcfIntrinsicValue",
        "dcfUpsidePct",
    ]
    if not rows:
        logger.warning("rank_opportunities: none of %d symbols could be scored", len(symbols))
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values("compositeScore", ascending=False)
        .reset_index(drop=True)
    )
