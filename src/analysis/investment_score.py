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

from src.analysis.business_model import FINANCIAL_KINDS
from src.analysis.dcf import run_dcf
from src.analysis.relative_valuation import PeerValuation, peer_valuation
from src.analysis.stock_analysis import analyze_stock
from src.analysis.valuation_summary import summarize_valuation
from src.config import (
    DEFAULT_MAX_PEERS,
    EQUITY_TO_ASSETS_BANDS,
    SCORE_RATING_BANDS,
    SCORE_WEIGHTS,
    VALUATION_SCORE_CAP,
)
from src.data.fmp_client import FMPClient, FMPError
from src.utils.logging import get_logger

logger = get_logger("investment_score")

# (threshold, score) bands, evaluated highest-threshold-first: value >= threshold wins.
_UPSIDE_BANDS = [(0.25, 100), (0.15, 85), (0.05, 70), (0.0, 55), (-0.10, 35), (-0.25, 15), (float("-inf"), 0)]
_MARGIN_BANDS = [(0.20, 100), (0.15, 80), (0.10, 60), (0.05, 40), (0.0, 20), (float("-inf"), 0)]
_ROE_BANDS = [(0.20, 100), (0.15, 80), (0.10, 60), (0.05, 40), (0.0, 20), (float("-inf"), 0)]
_GROWTH_BANDS = [(0.20, 100), (0.10, 80), (0.05, 60), (0.0, 40), (float("-inf"), 0)]
_INTEREST_COVERAGE_BANDS = [(10, 100), (5, 80), (2, 60), (1, 30), (float("-inf"), 0)]

# (threshold, score) bands, evaluated lowest-threshold-first: value <= threshold wins (lower is better).
_DEBT_TO_EQUITY_BANDS = [(0.3, 100), (0.6, 80), (1.0, 60), (2.0, 40), (float("inf"), 10)]
_CURRENT_RATIO_FLOOR_BANDS = [(0.75, 10), (1.0, 40), (1.5, 70), (float("inf"), 100)]


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


def _score_quality(stock: dict[str, Any]) -> dict[str, Any]:
    p = stock["profitability"]
    equity = stock["leverage"].get("totalEquity")
    negative_equity = equity is not None and equity <= 0
    roe = None if negative_equity else p["returnOnEquity"]  # ROE on negative equity is not meaningful
    roe_score = _band_score_higher_better(roe, _ROE_BANDS)
    margin_score = _band_score_higher_better(p["netProfitMargin"], _MARGIN_BANDS)

    valid_scores = [s for s in (roe_score, margin_score) if s is not None]
    score = round(sum(valid_scores) / len(valid_scores)) if valid_scores else None

    notes = []
    if negative_equity:
        notes.append("ROE n/m (negative equity)")
    elif p["returnOnEquity"] is not None:
        notes.append(f"ROE {p['returnOnEquity']:.1%}")
    if p["netProfitMargin"] is not None:
        notes.append(f"net margin {p['netProfitMargin']:.1%}")
    return {"score": score, "rationale": ", ".join(notes) if notes else "Insufficient data", "detail": p}


def _score_growth(stock: dict[str, Any]) -> dict[str, Any]:
    g = stock["growth"]
    revenue_score = _band_score_higher_better(g["revenueGrowthYoY"], _GROWTH_BANDS)
    income_score = _band_score_higher_better(g["netIncomeGrowthYoY"], _GROWTH_BANDS)

    valid_scores = [s for s in (revenue_score, income_score) if s is not None]
    score = round(sum(valid_scores) / len(valid_scores)) if valid_scores else None

    notes = []
    net_basis = g.get("revenueBasis") == "net of interest expense"
    if g["revenueGrowthYoY"] is not None:
        notes.append(f"{'net ' if net_basis else ''}revenue growth {g['revenueGrowthYoY']:.1%} YoY")
    if net_basis:
        notes.append("revenue net of interest expense (FMP reports it gross)")
    if g["netIncomeGrowthYoY"] is not None:
        notes.append(f"net income growth {g['netIncomeGrowthYoY']:.1%} YoY")
    elif g.get("priorNetIncomeNonPositive"):
        notes.append("net income growth n/m (prior-year loss)")
    return {"score": score, "rationale": ", ".join(notes) if notes else "Insufficient data", "detail": g}


def _score_financial_health(stock: dict[str, Any]) -> dict[str, Any]:
    leverage = stock["leverage"]
    if stock.get("businessModel") == "insurer":
        # Life insurers' total assets include separate-account (policyholder) assets, so
        # equity/assets is structurally low and says nothing about solvency (MET 3.8%).
        # The right metrics (RBC ratio, combined ratio) aren't available on FMP Starter.
        return {
            "score": None,
            "rationale": "not scored for insurers — solvency metrics (RBC ratio, combined ratio) unavailable",
            "detail": leverage,
        }
    if stock.get("captiveFinance"):
        # Consolidated current ratio, D/E and coverage all mix the finance arm's receivables,
        # short-term funding and interest with the operating business (DE current ratio 2.31).
        return {
            "score": None,
            "rationale": "not scored — captive-finance arm makes consolidated current ratio, D/E and "
            "coverage not meaningful for the operating business",
            "detail": leverage,
        }
    if stock.get("businessModel", "standard") in FINANCIAL_KINDS:
        # Deposit-/float-funded: current ratio, D/E and interest coverage are meaningless
        # (JPM scored 7/100 on them). Capital adequacy proxy: equity / total assets.
        eta = leverage.get("equityToAssets")
        score = _band_score_higher_better(eta, EQUITY_TO_ASSETS_BANDS)
        return {
            "score": score,
            "rationale": (
                f"equity/assets {eta:.1%} (financial: current ratio, D/E, coverage not applicable)"
                if eta is not None
                else "Insufficient data (equity/assets unavailable)"
            ),
            "detail": leverage,
        }
    equity = leverage.get("totalEquity")
    negative_equity = equity is not None and equity <= 0
    captive = bool(stock.get("captiveFinance"))
    interest_coverage = leverage["interestCoverageRatio"]
    # FMP reports exactly 0.0 (not null) when a company's interestExpense is
    # itself reported as 0 (e.g. Apple nets interest income/expense) — that's
    # a data gap, not a company with zero debt-service capacity, so treat it
    # as missing rather than penalizing it as the worst possible score.
    coverage_unreliable = interest_coverage == 0

    current_ratio_score = _band_score_lower_better(leverage["currentRatio"], _CURRENT_RATIO_FLOOR_BANDS)
    # Negative D/E (negative equity) must not land in the "lowest leverage" band (MCD scored 100).
    # Negative equity or a captive finance arm (Ford Credit's debt in consolidated D/E) makes
    # D/E not meaningful for the operating business.
    debt_equity_score = (
        None
        if negative_equity or captive
        else _band_score_lower_better(leverage["debtToEquityRatio"], _DEBT_TO_EQUITY_BANDS)
    )
    coverage_score = (
        None
        if coverage_unreliable or captive
        else _band_score_higher_better(interest_coverage, _INTEREST_COVERAGE_BANDS)
    )

    valid_scores = [s for s in (current_ratio_score, debt_equity_score, coverage_score) if s is not None]
    score = round(sum(valid_scores) / len(valid_scores)) if valid_scores else None

    notes = []
    if leverage["currentRatio"] is not None:
        notes.append(f"current ratio {leverage['currentRatio']:.2f}")
    if negative_equity:
        notes.append("debt/equity n/m (negative equity)")
    elif captive:
        notes.append("debt/equity n/m (includes captive-finance debt)")
    elif leverage["debtToEquityRatio"] is not None:
        notes.append(f"debt/equity {leverage['debtToEquityRatio']:.2f}")
    if captive:
        notes.append("interest coverage n/m (finance-arm interest booked in cost of sales)")
    elif coverage_unreliable:
        notes.append("interest coverage data unreliable (near-zero reported interest expense)")
    elif interest_coverage is not None:
        notes.append(f"interest coverage {interest_coverage:.1f}x")
    return {
        "score": score,
        "rationale": ", ".join(notes) if notes else "Insufficient data",
        "detail": leverage,
    }


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
    """The fundamental-quality rule, shared by the live scorecard and the backtester: business
    dimensions only (valuation is reported alongside, never blended in), each banded, missing
    dimensions dropped and the remaining weights renormalised. Returns (dimensions, composite
    0-100 or None when nothing is scoreable, rating or None)."""
    dimensions = {
        "quality": _score_quality(stock),
        "growth": _score_growth(stock),
        "financialHealth": _score_financial_health(stock),
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
