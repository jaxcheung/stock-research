# src/analysis/business_model.py
#
# One classification of how a company makes money, used to pick valuation
# methods and metrics that fit it (and refuse those that don't):
#
#   standard       industrial / services / tech — FCF-to-firm DCF and P/E apply
#   bank           deposit-funded banks — P/B vs ROE
#   lender         card / consumer / mortgage lenders, BDCs, mortgage REITs — P/B vs ROE
#   broker_dealer  balance-sheet brokers and wealth platforms (SCHW, IBKR, RJF) — P/B vs ROE
#   insurer        float-funded (subtype "life" or "pc") — P/B
#   asset_manager  fee businesses — P/E, no FCFF DCF
#   reit           equity REITs — P/FFO, no FCFF DCF
#
# Separately, `captive_finance` marks a non-financial company with a lending arm
# (Ford Credit, GM Financial, PACCAR Financial, John Deere Financial): its debt
# and receivables belong to a different business, so FCFF DCF and debt/equity
# are not meaningful for the consolidated company.
#
# Pure function, no I/O: callers pass the statements they already fetched. FMP
# files lenders (COF, SYF, AXP) and payment networks (V, MA, PYPL) under the same
# industry, so the industry name alone can't separate them — gross interest
# income / revenue can (see the config calibration notes). Config overrides are a
# disclosed backstop for what FMP's data can't show.

from dataclasses import dataclass
from typing import Any

from src.config import (
    BDC_INTEREST_INCOME_SHARE,
    BROKER_INDUSTRY_KEYWORDS,
    BUSINESS_MODEL_OVERRIDES,
    BUSINESS_MODEL_STRONG_KEYWORDS,
    BUSINESS_MODEL_WEAK_KEYWORDS,
    CAPTIVE_RECEIVABLES_SHARE,
    DCF_CAPTIVE_COD_GAP,
    DCF_CAPTIVE_DEBT_WEIGHT,
    DCF_RISK_FREE_RATE,
    LENDER_INTEREST_INCOME_SHARE,
    UNMAPPED_INTEREST_SHARE,
    UNMAPPED_LENDER_RECEIVABLES_SHARE,
)

KINDS = ("standard", "bank", "lender", "broker_dealer", "insurer", "asset_manager", "reit")
# Balance-sheet-funded: P/B, equity/assets (insurers: health not scored — see scorecard)
FINANCIAL_KINDS = ("bank", "lender", "broker_dealer", "insurer")

_LABELS = {
    "standard": "operating company",
    "bank": "bank",
    "lender": "lender",
    "broker_dealer": "broker-dealer / wealth platform",
    "insurer": "insurer",
    "asset_manager": "asset manager",
    "reit": "REIT",
}


@dataclass(frozen=True)
class BusinessModel:
    kind: str
    reason: str
    captive_finance: bool = False
    subtype: str | None = None  # insurers: "life" | "pc"
    override: str | None = None  # reason text when a config override was applied

    @property
    def label(self) -> str:
        if self.kind == "insurer" and self.subtype == "life":
            return "life insurer"
        return _LABELS[self.kind]

    @property
    def article(self) -> str:
        return "an" if self.label[0] in "aeiouAEIOU" else "a"

    @property
    def is_financial(self) -> bool:
        return self.kind in FINANCIAL_KINDS


def interest_income_share(income: dict[str, Any] | None) -> float | None:
    if not income:
        return None
    revenue, ii = income.get("revenue"), income.get("interestIncome")
    if not revenue or revenue <= 0 or ii is None:
        return None
    return float(ii) / float(revenue)


def receivables_share(balance: dict[str, Any] | None) -> float | None:
    if not balance:
        return None
    assets, recv = balance.get("totalAssets"), balance.get("netReceivables")
    if not assets or assets <= 0 or recv is None:
        return None
    return float(recv) / float(assets)


def _captive(
    profile: dict[str, Any], income: dict[str, Any] | None, balance: dict[str, Any] | None, rf: float
) -> str | None:
    """Reason string if the consolidated statements look like they include a finance arm."""
    rs = receivables_share(balance)
    if rs is not None and rs >= CAPTIVE_RECEIVABLES_SHARE:
        return f"receivables are {rs:.0%} of total assets (>= {CAPTIVE_RECEIVABLES_SHARE:.0%})"
    if not balance or not income:
        return None
    debt, cap = balance.get("totalDebt") or 0.0, profile.get("marketCap") or 0.0
    interest = income.get("interestExpense") or 0.0
    if debt > 0 and cap > 0 and interest > 0:
        kd, weight = interest / debt, debt / (debt + cap)
        if kd < rf - DCF_CAPTIVE_COD_GAP and weight > DCF_CAPTIVE_DEBT_WEIGHT:
            return (
                f"derived cost of debt {kd:.2%} is far below the risk-free rate ({rf:.2%}) while debt is "
                f"{weight:.0%} of capital — finance-arm interest booked outside interest expense"
            )
    return None


def classify(
    profile: dict[str, Any] | None,
    income: dict[str, Any] | None,
    balance: dict[str, Any] | None = None,
    risk_free_rate: float | None = None,
) -> BusinessModel:
    """Business model from FMP industry / sector, the latest annual income statement and
    (optionally) balance sheet. Without a balance sheet the receivables-based tests are
    skipped (the interest-income and keyword tests still apply)."""
    profile = profile or {}
    symbol = (profile.get("symbol") or "").upper()
    industry = profile.get("industry") or ""
    sector = profile.get("sector") or ""
    ind = industry.lower()
    rf = DCF_RISK_FREE_RATE if risk_free_rate is None else risk_free_rate

    ov = BUSINESS_MODEL_OVERRIDES.get(symbol)
    if ov:
        kind = ov.get("kind", "standard")
        return BusinessModel(
            kind,
            f"config override: {ov['reason']}",
            captive_finance=bool(ov.get("captive")),
            subtype=ov.get("subtype"),
            override=ov["reason"],
        )

    for keyword, kind in BUSINESS_MODEL_STRONG_KEYWORDS:
        if keyword.lower() in ind:
            subtype = None
            if kind == "insurer":
                subtype = "life" if "life" in ind else "pc"
            return BusinessModel(kind, f"FMP industry '{industry}'", subtype=subtype)

    if sector == "Financial Services":
        share = interest_income_share(income)
        if share is not None and share >= LENDER_INTEREST_INCOME_SHARE:
            pct = f"interest income is {share:.0%} of revenue"
            if any(k.lower() in ind for k in BROKER_INDUSTRY_KEYWORDS):
                if share >= BDC_INTEREST_INCOME_SHARE:
                    return BusinessModel("lender", f"{pct} (>= {BDC_INTEREST_INCOME_SHARE:.0%}) — lender/BDC")
                return BusinessModel("broker_dealer", f"{pct} — balance-sheet broker / wealth platform")
            return BusinessModel(
                "lender", f"{pct} (>= {LENDER_INTEREST_INCOME_SHARE:.0%}) — earns a loan spread"
            )
        rs = receivables_share(balance)
        if (
            (share is None or share < UNMAPPED_INTEREST_SHARE)
            and rs is not None
            and rs >= UNMAPPED_LENDER_RECEIVABLES_SHARE
        ):
            return BusinessModel(
                "lender",
                f"interest income not reported but receivables are {rs:.0%} of assets — a loan book",
            )
        for keyword, kind in BUSINESS_MODEL_WEAK_KEYWORDS:
            if keyword.lower() in ind:
                return BusinessModel(kind, f"FMP industry '{industry}'")
        return BusinessModel("standard", "Financial Services fee business (no material interest income)")

    captive = _captive(profile, income, balance, rf)
    if captive:
        return BusinessModel("standard", f"captive finance: {captive}", captive_finance=True)
    return BusinessModel("standard", "default")
