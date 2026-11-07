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
# - Eligible methods within ±TARGET_MAX_SINGLE_METHOD_GAP of the price are
#   blended into a price target (normal precision).
# - If every eligible method is beyond that gap AND they all point to the same
#   verdict, the verdict is reported with LOW precision and a range, not a
#   point target (PLTR: DCF -92% is out of band, peers -71% -> "Rich, low
#   precision", not "Indeterminate").
# - Eligible methods that point to different verdicts with none in band ->
#   "Indeterminate — methods disagree". Only non-credible methods ->
#   "Indeterminate — no credible method". No methods at all -> "n/a".
# - Divergence is checked across ALL eligible methods, so an out-of-band
#   method on the other side still raises the warning.
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
    precision: str  # "normal" | "low" | "none"
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

    if in_band:
        target = sum(m.value for m in in_band) / len(in_band)
        up = target / price - 1 if price else None
        text = "blended: average of DCF and peer-multiple" if len(in_band) == 2 else f"{in_band[0].name} only"
        return ValuationSummary(
            verdict_for(up) if up is not None else "n/a",
            "normal",
            target,
            None,
            up,
            text,
            methods,
            divergent,
            note,
            used=in_band,
        )
    if eligible and len(cats) == 1:
        vals = sorted(m.value for m in eligible)
        cat = next(iter(cats))
        nearest = min(eligible, key=lambda m: abs(m.upside or 0))
        return ValuationSummary(
            f"{cat} — low precision",
            "low",
            None,
            (vals[0], vals[-1]),
            nearest.upside,
            "every credible method is more than "
            f"{gap:.0%} from the price; range shown instead of a point target",
            methods,
            divergent,
            note,
            used=eligible,
        )
    if eligible:
        return ValuationSummary(
            "Indeterminate — methods disagree",
            "none",
            None,
            None,
            None,
            "methods disagree",
            methods,
            True,
            note,
        )
    if methods:
        return ValuationSummary(
            "Indeterminate — no credible method", "none", None, None, None, "no credible method", methods
        )
    return ValuationSummary("n/a", "none", None, None, None, "unavailable", methods)
