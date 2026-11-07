# src/analysis/business_model.py
#
# One classification of how a company makes money, used to pick valuation
# methods and metrics that fit it (and refuse those that don't):
#
#   standard       industrial / services / tech — FCF-to-firm DCF and P/E apply
#   bank           deposit-funded or balance-sheet intermediaries — P/B vs ROE
#   lender         card / consumer / mortgage lenders — P/B vs ROE
#   insurer        float-funded — P/B
#   asset_manager  fee businesses with balance-sheet investments — P/E, no FCFF DCF
#   reit           equity REITs — P/FFO, no FCFF DCF
#
# Pure function, no I/O: callers pass the profile and latest income statement
# they already fetched. FMP files lenders (COF, SYF, AXP) and payment networks
# (V, MA, PYPL) under the same industry, so the industry name alone can't
# separate them — gross interest income / revenue can (see config calibration).

from dataclasses import dataclass
from typing import Any

from src.config import BUSINESS_MODEL_INDUSTRY_KEYWORDS, LENDER_INTEREST_INCOME_SHARE

KINDS = ("standard", "bank", "lender", "insurer", "asset_manager", "reit")
FINANCIAL_KINDS = ("bank", "lender", "insurer")  # balance-sheet-funded: P/B, equity/assets

_LABELS = {
    "standard": "operating company",
    "bank": "bank / capital-markets intermediary",
    "lender": "lender",
    "insurer": "insurer",
    "asset_manager": "asset manager",
    "reit": "REIT",
}


@dataclass(frozen=True)
class BusinessModel:
    kind: str
    reason: str

    @property
    def label(self) -> str:
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


def classify(profile: dict[str, Any] | None, income: dict[str, Any] | None) -> BusinessModel:
    """Business model from FMP industry / sector and the latest annual income statement."""
    profile = profile or {}
    industry = profile.get("industry") or ""
    sector = profile.get("sector") or ""
    for keyword, kind in BUSINESS_MODEL_INDUSTRY_KEYWORDS:
        if keyword.lower() in industry.lower():
            return BusinessModel(kind, f"FMP industry '{industry}'")
    if sector == "Financial Services":
        share = interest_income_share(income)
        if share is not None and share >= LENDER_INTEREST_INCOME_SHARE:
            return BusinessModel(
                "lender",
                f"interest income is {share:.0%} of revenue (>= {LENDER_INTEREST_INCOME_SHARE:.0%}) — "
                "earns a spread on a loan book",
            )
    return BusinessModel("standard", "default")
