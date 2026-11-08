# src/analysis/stock_analysis.py

from typing import Any

from src.analysis.business_model import classify, revenue_pair, scorecard_revenue, yoy_growth
from src.analysis.data_quality import check_statements, latest_complete_balance
from src.analysis.dcf import live_risk_free_rate
from src.data.fmp_client import FMPClient, FMPError
from src.utils.logging import get_logger

logger = get_logger("stock_analysis")


def _growth(current: float | None, prior: float | None) -> float | None:
    return yoy_growth(current, prior)  # one rule for live and backtest (business_model.yoy_growth)


def analyze_stock(symbol: str, client: FMPClient | None = None, years: int = 5) -> dict[str, Any]:
    """Pull core fundamentals for a symbol and summarize valuation,
    profitability, growth, and leverage.
    """
    client = client or FMPClient()

    quote = client.get_quote(symbol)
    profile = client.get_profile(symbol)
    income = client.get_income_statement(symbol, limit=years)
    ratios = client.get_ratios(symbol, limit=years)
    key_metrics = client.get_key_metrics(symbol, limit=years)
    try:
        cash_rows = client.get_cash_flow(symbol, limit=years)
    except FMPError as exc:
        logger.warning("%s: cash-flow statement unavailable (%s)", symbol, exc)
        cash_rows = []
    try:
        balance = client.get_balance_sheet(symbol, limit=max(2, years))
    except FMPError as exc:
        logger.warning("%s: balance sheet unavailable (%s) — equity-based checks skipped", symbol, exc)
        balance = []
    latest_bs, bs_note = latest_complete_balance(balance)
    # Ratios computed on a stub balance sheet (RJF FY2026: "ROE 0.0%") are not real — use the
    # same prior period the balance-sheet fallback chose (data_quality.latest_complete_balance).
    period = 1 if (bs_note and balance and latest_bs is not balance[0]) else 0
    if bs_note:
        logger.warning("%s: %s", symbol, bs_note)
        if period:
            bs_note += " Profitability and leverage ratios use the same prior period."

    def _row_for(rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Row dated like the balance sheet in use (positional fallback if dates are absent)."""
        want = latest_bs.get("date")
        match = next((r for r in rows if want and r.get("date") == want), None)
        return match or (rows[period] if len(rows) > period else {})

    latest_ratios = _row_for(ratios)
    latest_metrics = _row_for(key_metrics)
    latest_income = income[0] if income else {}
    prior_income = income[1] if len(income) > 1 else {}

    # Foreign ADRs (e.g. Sony, Panasonic) trade in USD but report financials
    # in their home currency (JPY, CNY, KRW, ...). Any ratio mixing a USD
    # price with a local-currency per-share figure (P/E, P/B, EV/EBITDA, ...)
    # is meaningless without an FX conversion we don't do here — so those
    # get nulled out rather than silently returning a wrong number.
    # Profitability/growth/leverage are unaffected: they're ratios entirely
    # within the same reporting currency.
    quote_currency = profile.get("currency")
    reporting_currency = latest_income.get("reportedCurrency")
    currency_mismatch = bool(quote_currency and reporting_currency and quote_currency != reporting_currency)
    if currency_mismatch:
        logger.warning(
            "%s: quote currency (%s) != reporting currency (%s) — nulling price-derived valuation multiples",
            symbol,
            quote_currency,
            reporting_currency,
        )

    model = classify(profile, latest_income, latest_bs or None, risk_free_rate=live_risk_free_rate()[0])
    total_equity = latest_bs.get("totalStockholdersEquity")
    total_assets = latest_bs.get("totalAssets")
    equity_to_assets = total_equity / total_assets if total_equity is not None and total_assets else None

    # Banks / broker-dealers / card lenders: revenue net of interest expense (scorecard_revenue)
    rev_now, rev_prior, basis = revenue_pair(latest_income, prior_income or None, model.kind, model.subtype)
    net_basis = rev_now is not None and rev_now != latest_income.get("revenue")
    net_margin = latest_ratios.get("netProfitMargin")
    if net_basis and rev_now and latest_income.get("netIncome") is not None:
        net_margin = float(latest_income["netIncome"]) / rev_now
    prior_ni = prior_income.get("netIncome")

    revenue_history = [
        {
            "date": row.get("date"),
            "revenue": row.get("revenue"),
            # only years FMP reports GROSS get a net figure; others are None (shown as n/m), never
            # a reported figure presented as "net" (TFC FY2024)
            "netRevenue": (
                net
                if (net := scorecard_revenue(row, model.kind, model.subtype)) != row.get("revenue")
                else None
            ),
            "netIncome": row.get("netIncome"),
        }
        for row in income
    ]

    return {
        "symbol": symbol,
        "companyName": profile.get("companyName"),
        "sector": profile.get("sector"),
        "industry": profile.get("industry"),
        "price": quote.get("price"),
        "marketCap": profile.get("marketCap"),
        "quoteCurrency": quote_currency,
        "reportingCurrency": reporting_currency,
        "currencyMismatch": currency_mismatch,
        "dataNotes": [bs_note] if bs_note else [],
        "dataQuality": check_statements(income, balance, cash_rows, expected_years=years),
        "businessModel": model.kind,
        "businessModelReason": model.reason,
        "captiveFinance": model.captive_finance,
        "businessModelOverride": model.override,
        "valuation": {
            "peRatio": None if currency_mismatch else latest_ratios.get("priceToEarningsRatio"),
            "pbRatio": None if currency_mismatch else latest_ratios.get("priceToBookRatio"),
            "priceToSalesRatio": None if currency_mismatch else latest_ratios.get("priceToSalesRatio"),
            "evToEBITDA": None if currency_mismatch else latest_metrics.get("evToEBITDA"),
            "earningsYield": None if currency_mismatch else latest_metrics.get("earningsYield"),
        },
        "profitability": {
            "grossProfitMargin": latest_ratios.get("grossProfitMargin"),
            "operatingProfitMargin": latest_ratios.get("operatingProfitMargin"),
            "netProfitMargin": net_margin,
            "returnOnEquity": latest_metrics.get("returnOnEquity"),
            "returnOnAssets": latest_metrics.get("returnOnAssets"),
            "returnOnInvestedCapital": latest_metrics.get("returnOnInvestedCapital"),
        },
        "growth": {
            "revenueGrowthYoY": _growth(rev_now, rev_prior),
            "netIncomeGrowthYoY": _growth(latest_income.get("netIncome"), prior_ni),
            # basis "mixed" wins: growth is n/m because FMP switched presentation between the years
            "revenueBasis": "changed between years"
            if basis == "mixed"
            else "net of interest expense"
            if net_basis
            else "reported",
            "priorNetIncomeNonPositive": prior_ni is not None and prior_ni <= 0,
        },
        "leverage": {
            "currentRatio": latest_ratios.get("currentRatio"),
            "quickRatio": latest_ratios.get("quickRatio"),
            "debtToEquityRatio": latest_ratios.get("debtToEquityRatio"),
            "interestCoverageRatio": latest_ratios.get("interestCoverageRatio"),
            "totalEquity": total_equity,
            "equityToAssets": equity_to_assets,
        },
        "technical": {
            "priceAvg50": quote.get("priceAvg50"),
            "priceAvg200": quote.get("priceAvg200"),
            "yearHigh": quote.get("yearHigh"),
            "yearLow": quote.get("yearLow"),
        },
        "revenueHistory": revenue_history,
    }


def summarize_stock(symbol: str, client: FMPClient | None = None) -> str:
    """Render analyze_stock() output as a short markdown report."""
    a = analyze_stock(symbol, client=client)
    v, p, g, lev = a["valuation"], a["profitability"], a["growth"], a["leverage"]

    def pct(x: float | None) -> str:
        return f"{x * 100:.1f}%" if x is not None else "n/a"

    def num(x: float | None) -> str:
        return f"{x:.2f}" if x is not None else "n/a"

    lines: list[str] = [
        f"# {a['companyName']} ({a['symbol']})",
        f"{a['sector']} / {a['industry']} — price ${a['price']}, market cap ${a['marketCap']:,}"
        if a["marketCap"]
        else "",
        "",
        (
            f"⚠️ **Currency mismatch**: quote in {a['quoteCurrency']}, financials reported in "
            f"{a['reportingCurrency']} — valuation multiples below are unavailable (would require FX conversion)."
            if a["currencyMismatch"]
            else ""
        ),
        "## Valuation",
        f"- P/E: {num(v['peRatio'])}  |  P/B: {num(v['pbRatio'])}  |  P/S: {num(v['priceToSalesRatio'])}  |  EV/EBITDA: {num(v['evToEBITDA'])}",
        "",
        "## Profitability",
        f"- Gross margin: {pct(p['grossProfitMargin'])}  |  Operating margin: {pct(p['operatingProfitMargin'])}  |  Net margin: {pct(p['netProfitMargin'])}",
        f"- ROE: {pct(p['returnOnEquity'])}  |  ROA: {pct(p['returnOnAssets'])}  |  ROIC: {pct(p['returnOnInvestedCapital'])}",
        "",
        "## Growth (YoY)",
        f"- Revenue ({g.get('revenueBasis', 'reported')}): {pct(g['revenueGrowthYoY'])}  |  "
        f"Net income: {pct(g['netIncomeGrowthYoY'])}",
        "",
        "## Leverage & Liquidity",
        f"- Current ratio: {num(lev['currentRatio'])}  |  Quick ratio: {num(lev['quickRatio'])}  |  Debt/Equity: {num(lev['debtToEquityRatio'])}  |  Interest coverage: {num(lev['interestCoverageRatio'])}",
    ]
    return "\n".join(line for line in lines if line is not None)
