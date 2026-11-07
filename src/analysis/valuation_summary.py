# src/analysis/valuation_summary.py
#
# The single place that decides which valuation methods count and what the
# valuation verdict is. The research report's price target and the scorecard's
# valuation signal both read this, so they can never disagree (they did: SYF
# showed "Indeterminate" next to "valuation signal 100/100").
#
# Rules:
# - A method is ELIGIBLE unless its own engine flags it as not credible (DCF
#   outside the sanity band, peer P/B without a valid ROE adjustment).
# - Credible methods that point to different verdicts -> "Indeterminate —
#   methods disagree", whichever of them is inside the band.
# - Two agreeing credible methods, at least one within ±TARGET_MAX_SINGLE_METHOD_GAP
#   of the price -> point target from the in-band ones (normal precision, full score).
# - Two agreeing methods, both beyond the band -> "<verdict> — low precision" with
#   a range (score capped at VALUATION_SCORE_CAP).
# - One credible method within the band -> "<verdict> (single method)" (capped score).
# - One credible method beyond the band -> "Indicative only" (unscored): every such
#   case reviewed rested on a known weakness of that one method.
# - Only non-credible methods -> "Indeterminate — no credible method"; none -> "n/a".
# Valuation is a separate question from the fundamental-quality rating.

from dataclasses import dataclass, field
from typing import Any

from src.config import DCF_SANITY_BAND, TARGET_MAX_SINGLE_METHOD_GAP

ATTRACTIVE_UPSIDE = 0.15
RICH_UPSIDE = -0.10


def verdict_for(upside: float) -> str:
    if upside >= ATTRACTIVE_UPSIDE:
        return "Attractive (undervalued)"
    if upside >= RICH_UPSIDE:
        return "Fair value"
    return "Rich (overvalued)"


@dataclass
class MethodResult:
    name: str
    value: float
    upside: float | None
    eligible: bool
    in_band: bool
    reason: str | None = None  # why not used, if not used

    @property
    def category(self) -> str | None:
        return verdict_for(self.upside) if self.upside is not None else None


@dataclass
class ValuationSummary:
    verdict: str
    # "normal": >=2 agreeing credible methods, >=1 within the band (point target)
    # "single": one credible method within the band (point target, capped score)
    # "low": >=2 agreeing methods, all beyond the band (range, capped score)
    # "indicative": one credible method beyond the band (no verdict category, unscored)
    # "none": indeterminate or unavailable
    precision: str
    scoring: str  # "full" | "capped" | "none" — how investment_score may use it
    target: float | None
    target_range: tuple[float, float] | None
    upside: float | None
    method_text: str
    methods: list[MethodResult] = field(default_factory=list)
    divergent: bool = False
    divergence_note: str | None = None
    used: list[MethodResult] = field(default_factory=list)  # methods behind the verdict (scored)

    @property
    def indeterminate(self) -> bool:
        return self.verdict.startswith("Indeterminate")


def summarize_valuation(
    price: float | None, dcf: dict[str, Any] | None, peer: Any | None
) -> ValuationSummary:
    """`peer` is a relative_valuation.PeerValuation (duck-typed to avoid an import cycle)."""
    lo, hi = DCF_SANITY_BAND
    gap = TARGET_MAX_SINGLE_METHOD_GAP
    methods: list[MethodResult] = []

    def add(name: str, value: float, eligible: bool, reason: str | None) -> None:
        up = value / price - 1 if price else None
        in_band = up is not None and abs(up) <= gap
        if eligible and not in_band and up is not None:
            reason = f"{up:+.0%} from the price (beyond ±{gap:.0%})"
        methods.append(
            MethodResult(
                name, value, up, eligible, in_band, reason if (not eligible or not in_band) else None
            )
        )

    if dcf:
        breach = bool(dcf.get("sanityBandBreach"))
        add(
            "DCF",
            dcf["intrinsicValuePerShare"],
            not breach,
            f"outside the {lo:g}×–{hi:g}× price sanity band" if breach else None,
        )
    if peer is not None and getattr(peer, "implied_price", None) is not None:
        add(
            f"Peer {peer.method_label}",
            peer.implied_price,
            bool(peer.reliable),
            None if peer.reliable else "rests on an unadjusted multiple (see peer notes)",
        )

    eligible = [m for m in methods if m.eligible and m.upside is not None]
    in_band = [m for m in eligible if m.in_band]
    cats = {m.category for m in eligible}

    divergent, note = False, None
    if len(eligible) >= 2:
        vals = [m.value for m in eligible]
        if len(cats) > 1 or (max(vals) - min(vals)) / max(vals) > 0.30:
            divergent = True
            note = "; ".join(f"{m.name} {m.upside:+.0%}" for m in eligible)

    def build(**kw: Any) -> ValuationSummary:
        return ValuationSummary(methods=methods, divergent=divergent, divergence_note=note, **kw)

    # Credible methods pointing to different verdicts: no verdict, whichever one is in band
    # (GOOGL: peers +50% in band vs DCF -75% was "Attractive, 100/100").
    # Disagreement = opposite verdicts (one Attractive, one Rich) or a gap > 30 points.
    # Neighbouring categories close together (-9% vs -11%) agree — they straddle a band edge.
    ups = [m.upside for m in eligible if m.upside is not None]
    disagree = len(ups) >= 2 and (
        ("Attractive (undervalued)" in cats and "Rich (overvalued)" in cats) or max(ups) - min(ups) > 0.30
    )
    if disagree:
        for m in eligible:
            m.reason = m.reason or "credible methods disagree"
        return build(
            verdict="Indeterminate — methods disagree",
            precision="none",
            scoring="none",
            target=None,
            target_range=None,
            upside=None,
            method_text="credible methods point in different directions",
        )
    if len(eligible) >= 2 and in_band:
        target = sum(m.value for m in in_band) / len(in_band)
        up = target / price - 1 if price else None
        text = (
            "blended: average of DCF and peer-multiple"
            if len(in_band) == 2
            else f"{in_band[0].name} (the other method agrees but is beyond ±{gap:.0%})"
        )
        return build(
            verdict=verdict_for(up) if up is not None else "n/a",
            precision="normal",
            scoring="full",
            target=target,
            target_range=None,
            upside=up,
            method_text=text,
            used=in_band,
        )
    if len(eligible) >= 2:
        vals = sorted(m.value for m in eligible)
        nearest = min(eligible, key=lambda m: abs(m.upside or 0))
        return build(
            verdict=f"{verdict_for(sum(ups) / len(ups))} — low precision",
            precision="low",
            scoring="capped",
            target=None,
            target_range=(vals[0], vals[-1]),
            upside=nearest.upside,
            method_text=f"both methods agree but are more than {gap:.0%} from the price; range instead of a point target",
            used=eligible,
        )
    if len(eligible) == 1:
        m = eligible[0]
        if m.in_band:
            return build(
                verdict=f"{verdict_for(m.upside or 0)} (single method)",
                precision="single",
                scoring="capped",
                target=m.value,
                target_range=None,
                upside=m.upside,
                method_text=f"{m.name} only — no second method to confirm it",
                used=[m],
            )
        # One method, beyond the band: every case found rested on a known weakness of that
        # one method (SYF ROE scaling, FISV DCF, CAT's circular peer set) — indicative only.
        return build(
            verdict=f"Indicative only — {m.category}, single method beyond ±{gap:.0%}",
            precision="indicative",
            scoring="none",
            target=None,
            target_range=(m.value, m.value),
            upside=m.upside,
            method_text=f"{m.name} only, {m.upside:+.0%} from the price — not confirmed by a second method",
            used=[],
        )
    if methods:
        return build(
            verdict="Indeterminate — no credible method",
            precision="none",
            scoring="none",
            target=None,
            target_range=None,
            upside=None,
            method_text="no credible method",
        )
    return build(
        verdict="n/a",
        precision="none",
        scoring="none",
        target=None,
        target_range=None,
        upside=None,
        method_text="unavailable",
    )
