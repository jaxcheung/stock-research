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
    BDC_MIN_EQUITY_TO_ASSETS,
    BROKER_INDUSTRY_KEYWORDS,
    BUSINESS_MODEL_OVERRIDES,
    BUSINESS_MODEL_STRONG_KEYWORDS,
    BUSINESS_MODEL_WEAK_KEYWORDS,
    CAPTIVE_DEBT_TO_RECEIVABLES,
    CAPTIVE_INTEREST_INCOME_SHARE,
    CAPTIVE_NET_RECEIVABLES_SHARE,
    DCF_CAPTIVE_COD_GAP,
    DCF_CAPTIVE_DEBT_WEIGHT,
    DCF_RISK_FREE_RATE,
    LENDER_INTEREST_INCOME_SHARE,
    PLAIN_PB_SUBTYPES,
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
    subtype: str | None = None  # insurers: "life" | "pc"; lenders: "bdc" | "mreit" | None
    override: str | None = None  # reason text when a config override was applied

    @property
    def label(self) -> str:
        if self.kind == "insurer" and self.subtype == "life":
            return "life insurer"
        if self.subtype == "bdc":
            return "business development company (BDC)"
        if self.subtype == "mreit":
            return "mortgage REIT"
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


def scorecard_revenue(income: dict[str, Any] | None, kind: str, subtype: str | None = None) -> float | None:
    """Revenue on the basis the industry reports it. FMP's revenue for banks, broker-dealers and card
    lenders is GROSS (interest income + non-interest income), so in rising-rate years it grows with
    the deposit/funding cost it never keeps (JPM FY2023 +54% gross vs +21% net; 87% of banks got the
    top growth band). Net revenue = revenue − interest expense ties to company-reported figures (JPM
    $155bn, BAC $98.6bn, GS $46.3bn, AXP $60.5bn, COF $36.8bn). Applied only when interest income is
    clearly a component of revenue; BDCs and mortgage REITs (total investment income / net interest
    presentation) are left as reported."""
    if not income:
        return None
    revenue = income.get("revenue")
    if revenue is None:
        return None
    if kind not in ("bank", "lender", "broker_dealer") or subtype in PLAIN_PB_SUBTYPES:
        return float(revenue)
    ii, ie = income.get("interestIncome") or 0.0, income.get("interestExpense") or 0.0
    if ii > 0 and ie > 0 and revenue >= 0.98 * ii and revenue - ie > 0:
        return float(revenue) - float(ie)
    return float(revenue)


def financing_debt(balance: dict[str, Any] | None) -> float:
    """Debt EXCLUDING lease liabilities. Lease cost runs through operating expense (rent), not
    interest expense, so lease-heavy companies (MGM: $25.2bn of $31.6bn 'debt' is the VICI leases)
    showed a near-zero 'cost of debt' and were flagged as captive finance."""
    if not balance:
        return 0.0
    debt = balance.get("totalDebt") or 0.0
    leases = (balance.get("capitalLeaseObligationsCurrent") or 0.0) + (
        balance.get("capitalLeaseObligationsNonCurrent") or 0.0
    )
    return max(0.0, float(debt) - float(leases))


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
    """Reason string if the consolidated statements look like they include a finance arm
    (see config: three independent signals, each calibrated against false positives)."""
    if not balance:
        return None
    assets = balance.get("totalAssets") or 0.0
    recv = balance.get("netReceivables") or 0.0
    payables = balance.get("accountPayables") or 0.0
    debt, cap = financing_debt(balance), profile.get("marketCap") or 0.0
    if assets > 0 and recv > 0:
        net_share = (recv - payables) / assets
        if net_share >= CAPTIVE_NET_RECEIVABLES_SHARE and debt >= CAPTIVE_DEBT_TO_RECEIVABLES * recv:
            return (
                f"receivables net of payables are {net_share:.0%} of assets and debt is {debt / recv:.2f}× "
                "receivables — a debt-funded loan book, not trade credit"
            )
    if not income or debt <= 0 or cap <= 0:
        return None
    weight = debt / (debt + cap)
    if weight <= DCF_CAPTIVE_DEBT_WEIGHT:
        return None
    interest = income.get("interestExpense") or 0.0
    if interest > 0 and interest / debt < rf - DCF_CAPTIVE_COD_GAP:
        return (
            f"derived cost of debt {interest / debt:.2%} is far below the risk-free rate ({rf:.2%}) while debt "
            f"is {weight:.0%} of capital — finance-arm interest booked outside interest expense"
        )
    share = interest_income_share(income)
    if share is not None and share >= CAPTIVE_INTEREST_INCOME_SHARE:
        return (
            f"interest income is {share:.1%} of revenue while debt is {weight:.0%} of capital — "
            "an in-house lending business"
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
            elif keyword == "REIT - Mortgage":
                subtype = "mreit"
            return BusinessModel(kind, f"FMP industry '{industry}'", subtype=subtype)

    if sector == "Financial Services":
        share = interest_income_share(income)
        if share is not None and share >= LENDER_INTEREST_INCOME_SHARE:
            pct = f"interest income is {share:.0%} of revenue"
            if any(k.lower() in ind for k in BROKER_INDUSTRY_KEYWORDS):
                if share >= BDC_INTEREST_INCOME_SHARE:
                    # The 150% asset-coverage limit caps BDC debt at 2x equity, so a BDC's equity
                    # is at least ~1/3 of assets. Annuity insurers filed under Asset Management
                    # (CRBG: equity 3% of assets) also show ~100% "interest income".
                    eq = (balance or {}).get("totalStockholdersEquity")
                    ta = (balance or {}).get("totalAssets")
                    eta = eq / ta if eq is not None and ta else None
                    if eta is None or eta >= BDC_MIN_EQUITY_TO_ASSETS:
                        return BusinessModel(
                            "lender",
                            f"{pct} (>= {BDC_INTEREST_INCOME_SHARE:.0%}) — BDC / closed-end lender",
                            subtype="bdc",
                        )
                    return BusinessModel(
                        "lender",
                        f"{pct}, but equity is only {eta:.0%} of assets — not a BDC (leveraged lender or "
                        "spread-based insurer)",
                    )
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
