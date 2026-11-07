# src/analysis/dcf.py
#
# Two-stage discounted cash flow model:
#   1. Explicit projection of free cash flow for `projection_years`, grown at
#      `fcf_growth_rate` (defaults to historical FCF CAGR if not supplied).
#   2. Terminal value via the Gordon Growth Model at `terminal_growth_rate`.
# Discounted at a CAPM-derived WACC. Risk-free rate defaults to the live 10Y
# Treasury yield from FRED (falls back to a static assumption if unavailable);
# market risk premium remains a configurable assumption — override either with
# real-world figures as needed.

from typing import Any

from src.analysis.business_model import classify
from src.config import (
    DCF_CAPTIVE_COD_GAP,
    DCF_DEBT_SPREAD,
    DCF_DERIVED_GROWTH_BOUNDS,
    DCF_MARKET_RISK_PREMIUM,
    DCF_MAX_DERIVED_COST_OF_DEBT,
    DCF_PROJECTION_YEARS,
    DCF_RISK_FREE_RATE,
    DCF_SANITY_BAND,
    DCF_TERMINAL_GROWTH_RATE,
    DCF_TV_SHARE_FLAG,
    FRED_SERIES,
)
from src.data.fmp_client import FMPClient
from src.data.fred_client import FREDClient, FREDError
from src.utils.logging import get_logger

logger = get_logger("dcf")


def live_risk_free_rate() -> tuple[float, str]:
    """Live 10Y Treasury yield from FRED as the risk-free rate, with a
    labeled source so reports can disclose where the assumption came from.
    Falls back to the static config default if FRED is unavailable.
    """
    try:
        latest = FREDClient().get_latest(FRED_SERIES["treasury_10y"])
    except FREDError as exc:
        logger.warning("Could not fetch live risk-free rate from FRED, using default: %s", exc)
        return DCF_RISK_FREE_RATE, "default assumption (FRED unavailable)"

    if latest is None:
        return DCF_RISK_FREE_RATE, "default assumption (FRED unavailable)"
    return latest["value"] / 100, f"FRED 10Y Treasury yield as of {latest['date']}"


def _validate_dcf_inputs(
    projection_years: int,
    risk_free_rate: float,
    market_risk_premium: float,
    terminal_growth_rate: float,
    fcf_growth_rate: float | None,
    cost_of_debt: float | None,
) -> None:
    if not (1 <= projection_years <= 30):
        raise ValueError(f"projection_years must be between 1 and 30, got {projection_years}")
    if not (0 <= risk_free_rate <= 0.25):
        raise ValueError(f"risk_free_rate must be between 0% and 25%, got {risk_free_rate:.2%}")
    if not (0 <= market_risk_premium <= 0.20):
        raise ValueError(f"market_risk_premium must be between 0% and 20%, got {market_risk_premium:.2%}")
    if not (-0.05 <= terminal_growth_rate <= 0.10):
        raise ValueError(f"terminal_growth_rate must be between -5% and 10%, got {terminal_growth_rate:.2%}")
    if fcf_growth_rate is not None and not (-1.0 < fcf_growth_rate <= 2.0):
        raise ValueError(f"fcf_growth_rate must be between -100% and 200%, got {fcf_growth_rate:.2%}")
    if cost_of_debt is not None and not (0 <= cost_of_debt <= 0.5):
        raise ValueError(f"cost_of_debt must be between 0% and 50%, got {cost_of_debt:.2%}")


def _fcf_cagr(fcf_history: list[float]) -> float:
    """CAGR from oldest to newest FCF. Falls back to 5% if the series
    doesn't support a clean CAGR (e.g. negative or zero starting FCF).
    """
    if len(fcf_history) < 2 or fcf_history[0] <= 0 or fcf_history[-1] <= 0:
        logger.warning("FCF history unsuitable for CAGR, defaulting growth rate to 5%%")
        return 0.05
    periods = len(fcf_history) - 1
    return (fcf_history[-1] / fcf_history[0]) ** (1 / periods) - 1


def run_dcf(
    symbol: str,
    client: FMPClient | None = None,
    projection_years: int = DCF_PROJECTION_YEARS,
    risk_free_rate: float | None = None,
    market_risk_premium: float = DCF_MARKET_RISK_PREMIUM,
    terminal_growth_rate: float = DCF_TERMINAL_GROWTH_RATE,
    fcf_growth_rate: float | None = None,
    cost_of_debt: float | None = None,
) -> dict[str, Any]:
    if risk_free_rate is None:
        risk_free_rate, risk_free_rate_source = live_risk_free_rate()
    else:
        risk_free_rate_source = "user-specified"

    _validate_dcf_inputs(
        projection_years,
        risk_free_rate,
        market_risk_premium,
        terminal_growth_rate,
        fcf_growth_rate,
        cost_of_debt,
    )
    client = client or FMPClient()

    quote = client.get_quote(symbol)
    profile = client.get_profile(symbol)
    cash_flows = client.get_cash_flow(symbol, limit=5)
    income = client.get_income_statement(symbol, limit=1)
    balance_sheet = client.get_balance_sheet(symbol, limit=1)

    if not cash_flows or not income or not balance_sheet:
        raise ValueError(f"Insufficient financial data to run DCF for {symbol}")

    # Foreign ADRs trade in one currency (e.g. USD) but report financials in
    # another (e.g. JPY). Mixing them produces a meaningless intrinsic value
    # (market cap/price in USD, cash flows/equity value in local currency) —
    # refuse outright rather than silently returning a corrupted number.
    quote_currency = profile.get("currency")
    reporting_currency = income[0].get("reportedCurrency")
    if quote_currency and reporting_currency and quote_currency != reporting_currency:
        raise ValueError(
            f"{symbol}: quote is in {quote_currency} but financials are reported in "
            f"{reporting_currency} — DCF requires matching currencies (FX conversion not implemented)"
        )

    # Business model: an FCF-to-firm DCF assumes debt is financing and FCF is operating.
    # For banks/lenders/insurers deposits and float ARE operations and loan-book growth
    # sits outside "FCF" (JPM came out at -$1,139/share, AXP's "cost of debt" was its
    # deposit rate); for REITs property acquisitions sit outside FCF (O +176%).
    model = classify(profile, income[0], balance_sheet[0], risk_free_rate=risk_free_rate)
    if model.captive_finance:
        raise ValueError(
            f"{symbol}: consolidated statements include a captive-finance arm ({model.reason}) — its debt, "
            "interest and receivables growth belong to a lending business, so an FCF-to-firm DCF of the "
            "consolidated company is not meaningful without separating the finance segment"
        )
    if model.kind != "standard":
        alternative = {
            "broker_dealer": "P/B vs ROE, or a residual-income model",
            "bank": "P/B vs ROE, or a residual-income model",
            "lender": "P/B vs ROE, or a residual-income model",
            "insurer": "P/B, or a residual-income model",
            "asset_manager": "P/E on fee-related earnings",
            "reit": "P/FFO or AFFO per share",
        }[model.kind]
        raise ValueError(
            f"{symbol}: FCF-to-firm DCF is not meaningful for {model.article} {model.label} ({model.reason}) — "
            f"use {alternative}"
        )

    # Oldest -> newest for CAGR; FMP returns newest-first.
    fcf_history = [row["freeCashFlow"] for row in reversed(cash_flows) if row.get("freeCashFlow") is not None]
    if not fcf_history:
        raise ValueError(f"Insufficient free cash flow history to run DCF for {symbol}")
    base_fcf = fcf_history[-1]
    if base_fcf <= 0:
        raise ValueError(
            f"{symbol}: latest free cash flow is non-positive ({base_fcf:,.0f}) — "
            "growing it forward produces a meaningless (negative) intrinsic value"
        )
    growth_clamped = False
    if fcf_growth_rate is not None:
        growth_rate = fcf_growth_rate
    else:
        raw_growth = _fcf_cagr(fcf_history)
        lo, hi = DCF_DERIVED_GROWTH_BOUNDS
        growth_rate = min(max(raw_growth, lo), hi)
        growth_clamped = growth_rate != raw_growth
        if growth_clamped:
            logger.warning(
                "%s: historical FCF CAGR %.1f%% clamped to %.1f%% for projection",
                symbol,
                raw_growth * 100,
                growth_rate * 100,
            )

    latest_income = income[0]
    latest_bs = balance_sheet[0]

    pretax_income = latest_income.get("incomeBeforeTax")
    tax_expense = latest_income.get("incomeTaxExpense")
    effective_tax_rate = (
        max(0.0, min(tax_expense / pretax_income, 1.0))
        if pretax_income not in (None, 0) and tax_expense is not None
        else 0.21  # fallback: rough US statutory rate
    )

    total_debt = latest_bs.get("totalDebt") or 0.0
    cash = latest_bs.get("cashAndCashEquivalents") or 0.0
    net_debt = latest_bs.get("netDebt")
    if net_debt is None:
        net_debt = total_debt - cash

    market_cap = profile.get("marketCap")
    beta = profile.get("beta") or 1.0
    if not market_cap:
        raise ValueError(f"Missing market cap for {symbol}, cannot weight WACC")

    cost_of_equity = risk_free_rate + beta * market_risk_premium

    total_capital = market_cap + total_debt
    debt_weight = total_debt / total_capital
    derived_cost_of_debt: float | None = None
    cost_of_debt_floored = False
    if cost_of_debt is None:
        interest_expense = latest_income.get("interestExpense") or 0.0
        cost_of_debt = (
            interest_expense / total_debt
            if interest_expense and total_debt
            else (risk_free_rate + DCF_DEBT_SPREAD)
        )
        if cost_of_debt > DCF_MAX_DERIVED_COST_OF_DEBT:
            raise ValueError(
                f"{symbol}: derived cost of debt {cost_of_debt:.1%} (interest expense / total debt) is implausible "
                f"for an FCF-to-firm DCF (> {DCF_MAX_DERIVED_COST_OF_DEBT:.0%}) — typical of a lender whose "
                "interest expense funds its loan book; DCF not meaningful"
            )
        # A rate far below risk-free is a legacy fixed coupon (GOOGL 1.2%, XOM 1.4%), not the
        # marginal cost of new borrowing that WACC needs — replace it, and disclose.
        if cost_of_debt < risk_free_rate - DCF_CAPTIVE_COD_GAP:
            derived_cost_of_debt = cost_of_debt
            cost_of_debt = risk_free_rate + DCF_DEBT_SPREAD
            cost_of_debt_floored = True
            logger.info(
                "%s: derived cost of debt %.2f%% replaced by risk-free + spread %.2f%%",
                symbol,
                derived_cost_of_debt * 100,
                cost_of_debt * 100,
            )

    weight_equity = market_cap / total_capital
    weight_debt = debt_weight
    wacc = weight_equity * cost_of_equity + weight_debt * cost_of_debt * (1 - effective_tax_rate)

    if wacc <= terminal_growth_rate:
        raise ValueError(
            f"WACC ({wacc:.2%}) must exceed terminal growth rate "
            f"({terminal_growth_rate:.2%}) for the Gordon Growth terminal value to be valid"
        )

    projected_fcf = []
    pv_fcf = []
    fcf = base_fcf
    for year in range(1, projection_years + 1):
        fcf = fcf * (1 + growth_rate)
        pv = fcf / (1 + wacc) ** year
        projected_fcf.append(fcf)
        pv_fcf.append(pv)

    terminal_value = projected_fcf[-1] * (1 + terminal_growth_rate) / (wacc - terminal_growth_rate)
    pv_terminal_value = terminal_value / (1 + wacc) ** projection_years

    enterprise_value = sum(pv_fcf) + pv_terminal_value
    equity_value = enterprise_value - net_debt

    shares_outstanding = latest_income.get("weightedAverageShsOutDil") or latest_income.get(
        "weightedAverageShsOut"
    )
    if not shares_outstanding:
        raise ValueError(f"Missing diluted shares outstanding for {symbol}")

    if equity_value <= 0:
        raise ValueError(
            f"{symbol}: DCF equity value is non-positive (EV {enterprise_value:,.0f} < net debt "
            f"{net_debt:,.0f}) — no meaningful per-share value"
        )

    intrinsic_value_per_share = equity_value / shares_outstanding
    current_price = quote.get("price")
    upside_pct = (intrinsic_value_per_share - current_price) / current_price if current_price else None
    tv_share = pv_terminal_value / enterprise_value if enterprise_value > 0 else None
    lo, hi = DCF_SANITY_BAND
    ratio = intrinsic_value_per_share / current_price if current_price else None
    sanity_breach = ratio is not None and not (lo <= ratio <= hi)
    if sanity_breach:
        logger.warning(
            "%s: DCF value %.2fx the price is outside the %.1fx-%.1fx sanity band", symbol, ratio, lo, hi
        )

    return {
        "symbol": symbol,
        "assumptions": {
            "projectionYears": projection_years,
            "fcfGrowthRate": growth_rate,
            "fcfGrowthRateClamped": growth_clamped,
            "terminalGrowthRate": terminal_growth_rate,
            "riskFreeRate": risk_free_rate,
            "riskFreeRateSource": risk_free_rate_source,
            "marketRiskPremium": market_risk_premium,
            "beta": beta,
            "costOfEquity": cost_of_equity,
            "costOfDebt": cost_of_debt,
            "costOfDebtFloored": cost_of_debt_floored,
            "derivedCostOfDebt": derived_cost_of_debt,
            "effectiveTaxRate": effective_tax_rate,
        },
        "wacc": wacc,
        "baseFreeCashFlow": base_fcf,
        "projectedFreeCashFlows": projected_fcf,
        "presentValueOfFreeCashFlows": sum(pv_fcf),
        "terminalValue": terminal_value,
        "presentValueOfTerminalValue": pv_terminal_value,
        "enterpriseValue": enterprise_value,
        "netDebt": net_debt,
        "equityValue": equity_value,
        "sharesOutstanding": shares_outstanding,
        "intrinsicValuePerShare": intrinsic_value_per_share,
        "currentPrice": current_price,
        "upsidePct": upside_pct,
        "terminalValueShare": tv_share,
        "terminalValueShareFlag": tv_share is not None and tv_share > DCF_TV_SHARE_FLAG,
        # Reported, but never blended into a price target (research_report._price_target).
        "sanityBandBreach": sanity_breach,
        "businessModel": model.kind,
    }


def summarize_dcf(symbol: str, client: FMPClient | None = None, **kwargs: Any) -> str:
    """Render run_dcf() output as a short markdown report."""
    d = run_dcf(symbol, client=client, **kwargs)
    a = d["assumptions"]

    verdict = "undervalued" if d["upsidePct"] and d["upsidePct"] > 0 else "overvalued"

    return "\n".join(
        [
            f"# DCF Valuation: {symbol}",
            "",
            "## Assumptions",
            f"- Projection horizon: {a['projectionYears']} years",
            f"- FCF growth rate: {a['fcfGrowthRate']:.2%}"
            + (" (historical CAGR, clamped)" if a.get("fcfGrowthRateClamped") else "")
            + f"  |  Terminal growth rate: {a['terminalGrowthRate']:.2%}",
            f"- Risk-free rate: {a['riskFreeRate']:.2%} ({a['riskFreeRateSource']})",
            f"- WACC: {d['wacc']:.2%} (cost of equity {a['costOfEquity']:.2%}, cost of debt {a['costOfDebt']:.2%}, beta {a['beta']:.2f})",
            "",
            "## Valuation",
            f"- Enterprise value: ${d['enterpriseValue']:,.0f}"
            + (
                f" (terminal value {d['terminalValueShare']:.0%} of EV"
                + (" — ⚠️ dominated by the terminal assumption)" if d["terminalValueShareFlag"] else ")")
                if d["terminalValueShare"] is not None
                else ""
            ),
            f"- Less net debt: ${d['netDebt']:,.0f}",
            f"- Equity value: ${d['equityValue']:,.0f}",
            f"- Shares outstanding: {d['sharesOutstanding']:,.0f}",
            f"- **Intrinsic value/share: ${d['intrinsicValuePerShare']:.2f}** vs current price ${d['currentPrice']:.2f}",
            f"- Implied upside: {d['upsidePct']:.1%} ({verdict})" if d["upsidePct"] is not None else "",
        ]
    )
