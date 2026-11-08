# src/config.py

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent

# Logging
LOG_DIR = ROOT_DIR / "log"
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

# Financial Modeling Prep
FMP_API_KEY = os.environ.get("FMP_API_KEY")
FMP_BASE_URL = "https://financialmodelingprep.com/stable"

# DCF default assumptions (override per-call for sensitivity analysis)
DCF_PROJECTION_YEARS = 5
DCF_RISK_FREE_RATE = 0.04
DCF_MARKET_RISK_PREMIUM = 0.05
DCF_TERMINAL_GROWTH_RATE = 0.025
# Historical FCF CAGR used as the default growth rate is clamped to this range
# (a 5-year CAGR off a depressed base can be 50%+; compounding it is not a forecast).
DCF_DERIVED_GROWTH_BOUNDS = (-0.10, 0.15)
# A cost of debt derived from statements (interest expense / total debt) above this is
# implausible for an FCF-to-firm DCF — typically a lender whose interest expense funds
# a loan book (COF derived 31%). Secondary guard; the business-model check runs first.
DCF_MAX_DERIVED_COST_OF_DEBT = 0.15
# Captive-finance guard: derived cost of debt this far BELOW the risk-free rate while
# debt is more than this share of (debt + market cap) means total debt includes a finance
# arm whose interest sits in cost of sales (Ford 0.75%, GM 0.56% at ~75%/63% debt weight).
DCF_CAPTIVE_COD_GAP = 0.02
# A derived cost of debt this far below the risk-free rate is a legacy coupon, not the
# marginal cost of borrowing: it is replaced by risk-free + DCF_DEBT_SPREAD (disclosed).
DCF_DEBT_SPREAD = 0.01
DCF_CAPTIVE_DEBT_WEIGHT = 0.50
# Intrinsic value / price outside this band is reported but never blended into a target.
DCF_SANITY_BAND = (0.2, 5.0)
# Terminal value above this share of EV is flagged (the answer is mostly the TV assumption).
DCF_TV_SHARE_FLAG = 0.75

# Business-model classification (src/analysis/business_model.py). Order of tests:
# overrides -> REIT / bank / insurer keywords -> interest-income share -> unmapped-
# interest balance-sheet test -> weaker keywords -> standard. Keywords are substrings
# of the FMP industry name; first match wins ("REIT - Mortgage" is a lender).
BUSINESS_MODEL_STRONG_KEYWORDS: list[tuple[str, str]] = [
    ("REIT - Mortgage", "lender"),
    ("REIT", "reit"),
    ("Banks", "bank"),
    # Insurance BROKERS take no underwriting risk (AJG, BRO): fee businesses, judged like operating
    # companies ("standard" ends classification here: fiduciary-fund interest isn't lending).
    ("Insurance - Brokers", "standard"),
    ("Insurance", "insurer"),
]
# Financial Services companies are lenders when gross interest income is at least this
# share of revenue. Calibrated on the local store (FY2021-25): payment networks and
# processors <= 2.1% (V, MA 0; PYPL, GPN ~2%), lenders >= 20% (AXP 20-32%, SOFI 33-76%,
# COF ~80%, ALLY 81-111%, SYF 109-136%).
LENDER_INTEREST_INCOME_SHARE = 0.10
# Within Capital Markets / Investment Banking / Asset Management, an interest share at
# or above this means a lender or BDC (ARCC 97%); between LENDER_INTEREST_INCOME_SHARE
# and this, a balance-sheet broker-dealer or wealth platform (SCHW 56%, IBKR 41%, RJF 23%).
BDC_INTEREST_INCOME_SHARE = 0.80
# BDCs must hold equity >= this share of assets (150% asset coverage => debt <= 2x equity).
BDC_MIN_EQUITY_TO_ASSETS = 0.30
BROKER_INDUSTRY_KEYWORDS = ("Capital Markets", "Investment - Banking", "Asset Management")
# FMP sometimes leaves interest income unmapped (reported as ~0) for lenders (ENVA,
# SEZL, KLAR). When the share is below this AND receivables are at least
# UNMAPPED_LENDER_RECEIVABLES_SHARE of total assets, the company is treated as a lender.
# Receivables alone don't work: PYPL has 59% of assets in receivables but interest is mapped.
UNMAPPED_INTEREST_SHARE = 0.005
UNMAPPED_LENDER_RECEIVABLES_SHARE = 0.40
BUSINESS_MODEL_WEAK_KEYWORDS: list[tuple[str, str]] = [
    ("Financial - Mortgages", "lender"),
    ("Asset Management", "asset_manager"),
]
# Captive finance (non-financial companies with a lending arm). Any one of:
#  (1) DEBT-FUNDED receivables: (receivables - payables) >= CAPTIVE_NET_RECEIVABLES_SHARE of
#      assets AND debt >= CAPTIVE_DEBT_TO_RECEIVABLES x receivables. Calibrated 2026-11-07:
#      captives PCAR 44% / 0.72, DE 51% / 1.12, CNH 50% / 1.15; trade-receivables businesses
#      fail one leg: CHRW 25%, MAN 22%, ARW 8%, AVT 5%, KFRC 41% but 0.44, ABNB 31%.
#      (Receivables alone flagged freight brokers, staffing firms and distributors.)
#  (2) derived cost of debt more than DCF_CAPTIVE_COD_GAP below risk-free while debt exceeds
#      DCF_CAPTIVE_DEBT_WEIGHT of capital (F 0.75% / 77%, GM 0.56% / 64%).
#  (3) interest income >= CAPTIVE_INTEREST_INCOME_SHARE of revenue while debt exceeds
#      DCF_CAPTIVE_DEBT_WEIGHT of capital (KMX 7.2% / 71%; ABNB 5.7% but 2% debt weight).
# A low cost of debt alone is NOT a signal: GOOGL, META, XOM, MSFT derive < 2.5% from coupons.
CAPTIVE_NET_RECEIVABLES_SHARE = 0.40
CAPTIVE_DEBT_TO_RECEIVABLES = 0.60
CAPTIVE_INTEREST_INCOME_SHARE = 0.05
# Disclosed backstop for misfires the rules can't see from FMP data. Every applied
# override is shown in reports. Keys: kind (optional), captive (bool), subtype, reason.
BUSINESS_MODEL_OVERRIDES: dict[str, dict[str, Any]] = {
    "CAT": {"captive": True, "reason": "Cat Financial; finance receivables not mapped by FMP"},
    "HOG": {"captive": True, "reason": "Harley-Davidson Financial Services"},
    "CAR": {
        "captive": True,
        "reason": "vehicle-program (fleet) debt, which FMP files as lease obligations; its interest is "
        "booked as a vehicle cost, not interest expense",
    },
    "HGV": {"captive": True, "reason": "timeshare receivables funded by securitised debt"},
    "VAC": {"captive": True, "reason": "timeshare receivables funded by securitised debt"},
    "EQH": {
        "kind": "insurer",
        "subtype": "life",
        "peer_industries": ["Insurance - Life"],
        "reason": "life & annuity insurer filed as Asset Management",
    },
    "PFG": {
        "kind": "insurer",
        "subtype": "life",
        "peer_industries": ["Insurance - Life"],
        "reason": "life & retirement insurer filed as Asset Management",
    },
    "CRBG": {
        "kind": "insurer",
        "subtype": "life",
        "peer_industries": ["Insurance - Life"],
        "reason": "life & annuity insurer filed as Asset Management (spread income reads as interest income)",
    },
    "FCFS": {"kind": "lender", "reason": "pawn lender; loans not mapped to receivables or interest income"},
    "EZPW": {"kind": "lender", "reason": "pawn lender; loans not mapped to receivables or interest income"},
    "ECPG": {"kind": "lender", "reason": "debt buyer; portfolio income not mapped as interest income"},
}

# Extra FMP industries screened for "industry" peers of a business model, because FMP
# spreads that model across several industries (RJF/AMP sit in Asset Management, SCHW in
# Capital Markets, IBKR in Investment Banking). Peers must still match the model.
KIND_PEER_INDUSTRIES: dict[str, list[str]] = {
    "broker_dealer": [
        "Financial - Capital Markets",
        "Investment - Banking & Investment Services",
        "Asset Management",
    ],
    # Card/consumer lenders only: BDCs (also "lender", subtype "bdc") are matched on subtype
    # and screened in their own industry — pooling them with card lenders put ARCC at -48%.
    "lender": ["Financial - Credit Services"],
}

# Peer-multiple valuation (relative_valuation.peer_valuation): method by business model.
PEER_METHOD_BY_KIND = {
    "standard": "pe",
    "asset_manager": "pe",
    "bank": "pb",
    "lender": "pb",
    "insurer": "pb",
    "broker_dealer": "pb",
    "reit": "pffo",
}
PEER_TARGET_MIN_PEERS = 5  # fewer usable peers -> no peer-implied price
PEER_ROE_GAP_CAVEAT_PP = 5.0  # P/B re-rating caveat when subject vs peer ROE differ by more
PEER_EPS_TTM_DIVERGENCE = 2.0  # note when latest-FY vs TTM earnings differ by more than 2x
# Price target: a method more than this far from the price is shown but not blended;
# if no method qualifies the valuation verdict is "Indeterminate".
TARGET_MAX_SINGLE_METHOD_GAP = 0.50

# Industry/peer comparison
DEFAULT_MAX_PEERS = 10

# FMP request resilience
FMP_MAX_RETRIES = 3
FMP_RETRY_BACKOFF_SECONDS = 0.5  # doubles each retry: 0.5s, 1s, 2s
FMP_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

# Portfolio (transactions/PnL) — contains personal financial data, gitignored
PORTFOLIO_DB_PATH = ROOT_DIR / "data" / "portfolio.db"
# Float tolerance for share quantities (fractional shares accumulate binary rounding error)
PORTFOLIO_QTY_TOLERANCE = 1e-9

# Knowledge log (quotes/notes from books, fundamental analysis, Buffett, etc.)
KNOWLEDGE_DB_PATH = ROOT_DIR / "data" / "knowledge.db"

# FRED (Federal Reserve Economic Data) — macro/economic indicators
FRED_API_KEY = os.environ.get("FRED_API_KEY")

# LLM backend for the AI agents (src/llm). "claude_code" = headless `claude -p` on the user's
# Claude subscription (no API key); "anthropic_api" = Anthropic API (needs ANTHROPIC_API_KEY,
# billed separately; enables prompt-caching control and the Batch API).
LLM_BACKEND = os.environ.get("LLM_BACKEND") or "claude_code"
LLM_MODEL = os.environ.get("LLM_MODEL") or "claude-opus-5-5"
LLM_TIMEOUT_SECONDS = 600
LLM_USAGE_DB_PATH = ROOT_DIR / "data" / "llm_usage.db"  # gitignored (data/)
# SEC EDGAR (data-quality reconciliation, next round). SEC requires a User-Agent with a contact
# name and email on every request; EDGAR checks stay disabled until the user sets this.
SEC_USER_AGENT = os.environ.get("SEC_USER_AGENT")
FRED_BASE_URL = "https://api.stlouisfed.org/fred"
FRED_MAX_RETRIES = 3
FRED_RETRY_BACKOFF_SECONDS = 0.5
FRED_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

# Key FRED series used for the macro backdrop
FRED_SERIES = {
    "real_gdp_growth": "A191RL1Q225SBEA",  # Real GDP, % change SAAR, quarterly
    "cpi_index": "CPIAUCSL",  # CPI index level, monthly (YoY computed from this)
    "unemployment_rate": "UNRATE",  # %, monthly
    "fed_funds_rate": "FEDFUNDS",  # %, monthly
    "treasury_10y": "DGS10",  # %, daily
    "yield_curve_10y2y": "T10Y2Y",  # 10Y minus 2Y spread, %, daily
}

# Fundamental-quality scorecard weights (must sum to 1.0). Valuation is deliberately
# NOT in the composite: the rating answers "how good is the business", the valuation
# verdict answers "is the price right" (CLAUDE.md Rules). Weights keep the original
# relative proportions of the four business dimensions (0.25 : 0.20 : 0.15 : 0.05).
SCORE_WEIGHTS = {
    "quality": 0.38,
    "growth": 0.31,
    "financialHealth": 0.23,
    "momentum": 0.08,
}

# Financial health for banks/lenders/insurers: equity / total assets (current ratio,
# D/E and interest coverage are meaningless for deposit- or float-funded balance sheets).
EQUITY_TO_ASSETS_BANDS = [(0.12, 100), (0.10, 80), (0.08, 60), (0.06, 30), (float("-inf"), 10)]

# ---------------------------------------------------------------- scoring templates
# One scoring template per business model (src/analysis/templates.py). A template says WHICH
# metrics are scored in each scorecard dimension, which dimensions are unscored by design (and
# why), which indicators to show, which valuation methods apply, and which industry metrics FMP
# Starter does not provide. Live scorecard, research report, point-in-time backtest and (later)
# AI agents all read the same template — no consumer picks its own indicators.
#
# Bands: (threshold, score). mode "higher": first threshold the value is >= wins (highest first).
# mode "lower": first threshold the value is <= wins (lowest first). The generic bands are the
# original scorecard's; the bank bands (ROA, efficiency) are calibrated so a typical bank lands
# mid-range (cross-section of FY2023-25 statements, not returns).
SCORECARD_METRIC_BANDS: dict[str, dict[str, Any]] = {
    # generic (original scorecard)
    "roe": {
        "mode": "higher",
        "bands": [(0.20, 100), (0.15, 80), (0.10, 60), (0.05, 40), (0.0, 20), (float("-inf"), 0)],
    },
    "netMargin": {
        "mode": "higher",
        "bands": [(0.20, 100), (0.15, 80), (0.10, 60), (0.05, 40), (0.0, 20), (float("-inf"), 0)],
    },
    "revenueGrowth": {
        "mode": "higher",
        "bands": [(0.20, 100), (0.10, 80), (0.05, 60), (0.0, 40), (float("-inf"), 0)],
    },
    "netIncomeGrowth": {
        "mode": "higher",
        "bands": [(0.20, 100), (0.10, 80), (0.05, 60), (0.0, 40), (float("-inf"), 0)],
    },
    "epsGrowth": {
        "mode": "higher",
        "bands": [(0.20, 100), (0.10, 80), (0.05, 60), (0.0, 40), (float("-inf"), 0)],
    },
    "currentRatio": {"mode": "lower", "bands": [(0.75, 10), (1.0, 40), (1.5, 70), (float("inf"), 100)]},
    "debtToEquity": {
        "mode": "lower",
        "bands": [(0.3, 100), (0.6, 80), (1.0, 60), (2.0, 40), (float("inf"), 10)],
    },
    "interestCoverage": {
        "mode": "higher",
        "bands": [(10, 100), (5, 80), (2, 60), (1, 30), (float("-inf"), 0)],
    },
    "equityToAssets": {"mode": "higher", "bands": EQUITY_TO_ASSETS_BANDS},
    # banks: ROA (US bank median ~1.0-1.1%), efficiency ratio (median ~60%; lower is better)
    "roa": {
        "mode": "higher",
        "bands": [(0.015, 100), (0.012, 80), (0.010, 60), (0.0075, 40), (0.005, 20), (float("-inf"), 0)],
    },
    "efficiencyRatio": {
        "mode": "lower",
        "bands": [(0.50, 100), (0.55, 80), (0.62, 60), (0.68, 40), (0.78, 20), (float("inf"), 0)],
    },
}

SCORECARD_TEMPLATES: dict[str, dict[str, Any]] = {
    "standard": {
        "label": "General (operating companies)",
        "kinds": ["standard", "asset_manager"],
        "dimensions": {
            "quality": ["roe", "netMargin"],
            "growth": ["revenueGrowth", "netIncomeGrowth"],
            "financialHealth": ["currentRatio", "debtToEquity", "interestCoverage"],
        },
        "why": "Operating businesses: returns on equity and margins, top- and bottom-line growth, "
        "liquidity, leverage and debt service.",
        "valuation": ["DCF (FCF, two-stage)", "Peer P/E (Σcap ÷ Σearnings)"],
        "display": [
            "grossMargin",
            "operatingMargin",
            "fcfMargin",
            "roic",
            "netDebtToEbitda",
            "pe",
            "evToEbitda",
        ],
        "unavailable": [],
    },
    "bank": {
        "label": "Banks",
        "kinds": ["bank"],
        "dimensions": {
            "quality": ["roe", "roa", "efficiencyRatio"],
            "growth": ["revenueGrowth", "epsGrowth"],
            "financialHealth": ["equityToAssets"],
        },
        "why": "Banks are balance-sheet businesses: profitability is ROE and ROA, cost discipline is the "
        "efficiency ratio (operating expenses ÷ net revenue), growth is NET revenue (after interest "
        "expense) and EPS, and capital strength is proxied by equity ÷ assets. Net margin, current ratio, "
        "D/E and interest coverage are meaningless for a deposit-funded balance sheet.",
        "valuation": ["ROE-adjusted peer P/B", "DCF not applicable (deposits are raw material, not capital)"],
        "display": [
            "netInterestShare",
            "efficiencyRatio",
            "roa",
            "pTangibleBook",
            "dividendYield",
            "payoutRatio",
        ],
        "unavailable": [
            "CET1 / Tier 1 capital ratio (regulatory filings, not in FMP Starter) — equity ÷ assets used instead",
            "Net interest margin on average earning assets",
            "Non-performing loans, net charge-offs, allowance coverage",
            "Deposit mix and deposit beta",
        ],
    },
    "reit": {
        # INTERIM: the general rules (unchanged from before templates). A REIT-specific template needs
        # NAREIT FFO, which FMP Starter can't reconstruct: impairments, property-sale gains and
        # discontinued operations aren't separable, and "other income" nets interest expense — a
        # proxy built from them produced wrong-direction FFO growth (BXP, MAC) and a 120% FFO
        # margin (RYN). Redesign pending (see CLAUDE.md milestones).
        "label": "REITs (interim: general rules)",
        "kinds": ["reit"],
        "dimensions": {
            "quality": ["roe", "netMargin"],
            "growth": ["revenueGrowth", "netIncomeGrowth"],
            "financialHealth": ["currentRatio", "debtToEquity", "interestCoverage"],
        },
        "why": "Interim: REITs are scored on the general rules. GAAP earnings understate REITs (real-estate "
        "depreciation), so quality and growth here are conservative; a REIT template based on FFO needs data "
        "FMP Starter does not provide.",
        "valuation": ["Peer P/FFO (approximate FFO = net income + D&A)", "DCF not applicable"],
        "display": ["ffoMargin", "pFfo", "dividendYield", "payoutRatio", "debtToAssets", "netDebtToEbitda"],
        "unavailable": [
            "NAREIT FFO / AFFO (impairments, gains on sale, discontinued operations not separable)",
            "Same-store NOI growth, occupancy, lease expiries",
            "Net asset value (NAV) per share",
            "EBITDAre and lease-adjusted leverage",
        ],
    },
    "insurer": {
        # INTERIM: the pre-template insurer rules. The ROE-only / EPS-growth template over-weighted
        # insurers 3x in the picks (hard P&C market, catastrophe recoveries, equity mark-to-market).
        # Redesign pending: book-value-per-share growth, multi-year ROE, debt/capital.
        "label": "Insurers (interim rules)",
        "kinds": ["insurer"],
        "dimensions": {
            "quality": ["roe", "netMargin"],
            "growth": ["revenueGrowth", "netIncomeGrowth"],
        },
        "unscored": {
            "financialHealth": "not scored for insurers — solvency metrics (RBC ratio, combined ratio) are not "
            "available on FMP Starter, and equity ÷ assets is distorted by separate-account assets",
        },
        "why": "Interim rules: return on equity and net margin, revenue and earnings growth. An insurer-specific "
        "template (book value growth, multi-year ROE, debt/capital) is pending.",
        "valuation": ["ROE-adjusted peer P/B", "DCF not applicable"],
        "display": ["roe", "financialLeverage", "debtToCapital", "pb", "dividendYield"],
        "unavailable": [
            "Combined ratio, loss ratio, reserve development (P&C)",
            "Risk-based capital (RBC) / solvency ratio",
            "Embedded value, new-business margin (life)",
            "Earnings ex investment mark-to-market (equity-heavy insurers: BRK, MKL)",
        ],
    },
    "lender_broker": {
        "label": "Lenders & broker-dealers",
        "kinds": ["lender", "broker_dealer"],
        "dimensions": {
            "quality": ["roe"],
            "growth": ["revenueGrowth", "netIncomeGrowth"],
            "financialHealth": ["equityToAssets"],
        },
        "why": "Balance-sheet financials without a bank-specific template yet: ROE (margins aren't comparable "
        "without cost of goods sold), net revenue and earnings growth, equity ÷ assets.",
        "valuation": [
            "ROE-adjusted peer P/B (plain P/B ≈ P/NAV for BDCs and mortgage REITs)",
            "DCF not applicable",
        ],
        "display": ["roe", "roa", "equityToAssets", "pb", "dividendYield"],
        "unavailable": ["Credit quality (charge-offs, non-accruals)", "Regulatory capital"],
    },
}

# Peer P/B ROE adjustment safeguards: peers with negative TTM earnings are excluded from
# the P/B and ROE aggregates; peer ROE must exceed long-run growth by at least this; and the
# subject's ROE must lie within the range of peer ROEs (interpolation, not extrapolation —
# a bound on the adjustment factor measured the wrong thing: it hid AXP, admitted SYF).
PEER_ROE_MIN_SPREAD = 0.04
# Tolerance around the peer ROE range: industry leaders sit just above their peers (JPM
# 17.9% vs big-bank peers 8.3-13.7%, SCHW +1.5pp, MET +1.5pp) and a strict range rejected
# every one of them. Beyond range + tolerance the P/B re-rating is "unreliable" (AXP).
PEER_ROE_RANGE_TOLERANCE = 0.05
# Lender subtypes valued on plain P/B (≈ price / NAV — the industry-standard metric), with no
# ROE adjustment: GAAP ROE includes mark-to-market for mortgage REITs, and a BDC's value is
# its NAV and dividend, not a card lender's ROE.
PLAIN_PB_SUBTYPES = ("bdc", "mreit")

# Valuation signal scores (0-100) are clamped to this range unless the verdict rests on at
# least two agreeing methods with one inside the band (no extreme scores on thin evidence).
VALUATION_SCORE_CAP = (15, 85)

# Data quality: a latest-period balance sheet whose total assets are below this fraction of
# the prior period's (or with zero equity AND zero liabilities) is a stub row (RJF FY2026:
# 3 non-zero fields) — analyses fall back to the prior period and say so.
STUB_STATEMENT_ASSET_RATIO = 0.10
# Statement checks (data_quality.check_statements) — surfaced as issues, never auto-corrected.
DQ_IDENTITY_TOLERANCE = 0.01  # accounting identities may differ by at most 1% (of assets / CFO / NI)
DQ_REVENUE_JUMP = 3.0  # YoY revenue x3 or /3: flag (M&A, restatement, or a units error)
DQ_SHARE_CHANGE = 0.20  # diluted share count moving >20% YoY: split, merger or large buyback?
DQ_MIN_PERIOD_DAYS = 300  # an "annual" period shorter than this is a transition period (LHX 186d)
DQ_MAX_GAP_DAYS = 400  # consecutive annual periods further apart than this = a missing year
DQ_TIEOUT_WARN = 0.10  # income vs cash-flow net income further apart than this = warning, not note
DQ_UNMAPPED_MATERIALITY = 0.01  # field material last year (>1% of revenue/assets) but 0 now

# Composite score (0-100) -> rating thresholds
SCORE_RATING_BANDS = [
    (80, "Strong Buy"),
    (65, "Buy"),
    (45, "Hold"),
    (30, "Sell"),
    (0, "Strong Sell"),
]

# FRED keyless fallback: when FRED_API_KEY is unset, series are pulled from
# FRED's public CSV download (no key, full history, same series IDs).
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

# World Bank Indicators API — keyless, annual, global country coverage.
WORLD_BANK_BASE_URL = "https://api.worldbank.org/v2"
WORLD_BANK_MAX_RETRIES = 3
WORLD_BANK_RETRY_BACKOFF_SECONDS = 0.5
WORLD_BANK_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

# Local columnar store: Parquet files queried with DuckDB (data/store/, gitignored).
STORE_DIR = ROOT_DIR / "data" / "store"
STORE_STATEMENT_MAX_AGE_HOURS = 24 * 7  # bulk sync skips company files refreshed more recently
SYNC_DEFAULT_YEARS = 10  # annual statements per company (Starter tier returns up to 10)
SYNC_UNIVERSE_MIN_CAP = 10_000_000_000  # default bulk universe: US-listed companies >= $10B

# ---------------------------------------------------------------- backtesting (src/backtest)
# Universe: US exchanges only (NYSE / NASDAQ / NYSE American), long-only, mid + large caps.
BACKTEST_UNIVERSE_NAME = "us_2bn"  # store universe snapshot synced with --min-cap 2e9
BACKTEST_MIN_CAP = 2_000_000_000  # point-in-time membership floor at each rebalance date
BACKTEST_PRICE_START = "2010-01-01"  # FMP Starter serves dividend-adjusted EOD back to at least 2010
# Top-down layer and benchmarks: US-listed market and sector index ETFs.
BACKTEST_MARKET_ETFS = ["SPY", "RSP", "QQQ", "MDY", "IWM"]
BACKTEST_SECTOR_ETFS = {
    "XLK": "Technology",
    "XLF": "Financial Services",
    "XLV": "Healthcare",
    "XLY": "Consumer Cyclical",
    "XLP": "Consumer Defensive",
    "XLE": "Energy",
    "XLI": "Industrials",
    "XLB": "Basic Materials",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
    "XLC": "Communication Services",
}
BACKTEST_BENCHMARK = "SPY"
BACKTEST_COST_BPS = 10.0  # one-way transaction cost per unit of turnover (commission + spread)
BACKTEST_MAX_WEIGHT = 0.05  # long-only position cap (stocks); ETF strategies use BACKTEST_ETF_MAX_WEIGHT
BACKTEST_ETF_MAX_WEIGHT = 0.40
BACKTEST_SECTOR_CAP = 0.30  # max weight per sector in stock strategies
BACKTEST_TOP_N = 50  # stocks held by factor strategies
BACKTEST_FILING_LAG_DAYS = 1
# ...and never earlier than this many days after the fiscal year end, whatever the filing date
# says: FMP shows filingDate == period end for some pre-listing/predecessor rows (CEG 2016-20),
# which would otherwise leak a full year of fundamentals. 10-K deadlines are 60-90 days.
BACKTEST_MIN_REPORT_LAG_DAYS = 60
BACKTEST_RISK_FREE_SERIES = "fred:TB3MS"  # 3-month T-bill, % p.a., monthly
BACKTEST_MIN_UNIVERSE = 100  # a stock strategy needs at least this many eligible stocks at a rebalance
BACKTEST_MIN_HISTORY_MONTHS = 13  # momentum 12-1 needs 13 month-ends of prices  # a statement is usable from the trading day AFTER its filing/acceptance
BACKTEST_WALK_FORWARD_TEST_FRACTION = 0.3  # last 30% of months held out as out-of-sample
BACKTEST_BOOTSTRAP_BLOCK = 6  # months per block in the stationary block bootstrap
BACKTEST_BOOTSTRAP_SAMPLES = 2000
# Minimum months for an evaluation (HAC alpha, bootstrap and deflated Sharpe need >= 24).
BACKTEST_MIN_EVAL_MONTHS = 24
# Liquidity screen for stock-universe membership: median daily dollar volume (close x volume)
# over the last BACKTEST_LIQUIDITY_WINDOW sessions. Keeps thinly traded predecessor listings
# out (GLXY's pre-Nasdaq OTC line traded 3-134 shares a day but cleared the $2bn cap floor).
BACKTEST_MIN_DOLLAR_VOLUME = 5e6
# Scorecard backtests start at the first month where at least this share of eligible stocks has a
# GROWTH score (it needs two filed full years): before that, the "scorecard" is a 3-dimension model
# (quality 0.38/0.69 instead of 0.38); in testing that first year produced all of scorecard_top's
# apparent edge over SPY.
BACKTEST_SCORECARD_MIN_GROWTH_COVERAGE = 0.8
# Version of the fundamental-scorecard RULES (bands, definitions). Bump it whenever a rule changes:
# it is part of the trial-ledger key, so a rule revised after seeing backtest results counts as a
# new trial in the deflated Sharpe instead of silently replacing the old one.
#   1 = original; 2 = net revenue for banks, growth-base rule, lease-excluded captive debt,
#       ROE-only quality for bank-type models (2026-11-08)
#   3 = first template draft incl. REIT FFO and ROE-only insurer templates (withdrawn after review:
#       wrong FFO arithmetic, insurer tilt) — its backtest trials stay in the ledger (2026-11-08)
#   4 = bank template (ROE, ROA, efficiency ratio, net revenue + EPS growth, equity/assets);
#       insurance brokers scored as operating companies; REITs and insurers on interim
#       (pre-template) rules (2026-11-08)
SCORECARD_RULES_VERSION = 4
BACKTEST_LIQUIDITY_WINDOW = 63
# Random-pick placebo: this many seeds, GROSS of costs, judged against the equal-weight universe.
# Expect about 5% of seeds "significant" at the 5% level; materially more means a biased pipeline.
BACKTEST_PLACEBO_SEEDS = 20
# Persistent record of every strategy configuration ever evaluated, so the deflated Sharpe's
# trial count can't be gamed by running strategies one at a time (gitignored, under data/).
BACKTEST_TRIAL_LEDGER_PATH = ROOT_DIR / "data" / "backtest_trials.db"

# 52/53-week fiscal years ending within this many days into January belong to the prior year
FISCAL_YEAR_END_GRACE_DAYS = 14
PANEL_MIN_COMPANIES = 5  # suppress sector-year medians computed from fewer companies
# CAGR from a tiny base (shells, recent spin-ups, acquisition-built companies) is not meaningful
LEADERBOARD_MIN_START_REVENUE = 500_000_000  # in reporting currency

# FMP Starter allows ~300 calls/minute; stay safely under it. Cache hits are free.
FMP_MAX_CALLS_PER_MINUTE = 250

# Local response cache (data/cache/, gitignored). TTLs in seconds, keyed by
# how fast the underlying data actually changes. 0 disables caching.
CACHE_DIR = ROOT_DIR / "data" / "cache"
CACHE_ENABLED = os.environ.get("CACHE_ENABLED", "1") != "0"
CACHE_TTL_SECONDS = {
    "quote": 5 * 60,  # live prices
    "statements": 24 * 3600,  # annual/quarterly financial statements, ratios, metrics
    "market": 6 * 3600,  # sector/industry PE snapshots, screener, price history
    "macro": 12 * 3600,  # FRED / World Bank series
    "default": 3600,
}

# Macro series catalog for the multi-variable Macro Lab. Each entry is a
# (source, id) the series loader understands; add new series here, not in code.
#   source "fred": FRED series ID       source "worldbank": "<ISO3>:<indicator>"
#   unit "pct" = already in percentage points; "index"/"level" = raw level.
MACRO_SERIES_CATALOG: dict[str, dict[str, str]] = {
    # --- China ---
    "cn_interbank_3m": {
        "source": "fred",
        "id": "IR3TIB01CNM156N",
        "label": "China 3M interbank rate",
        "unit": "pct",
        "freq": "M",
        "region": "China",
    },
    "cn_discount_rate": {
        "source": "fred",
        "id": "INTDSRCNM193N",
        "label": "China central bank discount rate",
        "unit": "pct",
        "freq": "M",
        "region": "China",
    },
    "cn_lending_rate": {
        "source": "worldbank",
        "id": "CHN:FR.INR.LEND",
        "label": "China bank lending rate",
        "unit": "pct",
        "freq": "A",
        "region": "China",
    },
    "cn_cpi_yoy": {
        "source": "fred",
        "id": "CPALTT01CNM659N",
        "label": "China CPI inflation (YoY)",
        "unit": "pct",
        "freq": "M",
        "region": "China",
    },
    "cn_exports": {
        "source": "fred",
        "id": "XTEXVA01CNM667S",
        "label": "China exports (USD, SA)",
        "unit": "level",
        "freq": "M",
        "region": "China",
    },
    "cn_gdp_growth": {
        "source": "worldbank",
        "id": "CHN:NY.GDP.MKTP.KD.ZG",
        "label": "China real GDP growth",
        "unit": "pct",
        "freq": "A",
        "region": "China",
    },
    "usd_cny": {
        "source": "fred",
        "id": "DEXCHUS",
        "label": "CNY per USD",
        "unit": "level",
        "freq": "D",
        "region": "China",
    },
    # --- United States ---
    "us_fed_funds": {
        "source": "fred",
        "id": "FEDFUNDS",
        "label": "US Fed funds rate",
        "unit": "pct",
        "freq": "M",
        "region": "US",
    },
    "us_10y": {
        "source": "fred",
        "id": "DGS10",
        "label": "US 10Y Treasury yield",
        "unit": "pct",
        "freq": "D",
        "region": "US",
    },
    "us_curve_10y2y": {
        "source": "fred",
        "id": "T10Y2Y",
        "label": "US yield curve (10Y-2Y)",
        "unit": "pct",
        "freq": "D",
        "region": "US",
    },
    "us_cpi": {
        "source": "fred",
        "id": "CPIAUCSL",
        "label": "US CPI index",
        "unit": "index",
        "freq": "M",
        "region": "US",
    },
    "us_unemployment": {
        "source": "fred",
        "id": "UNRATE",
        "label": "US unemployment rate",
        "unit": "pct",
        "freq": "M",
        "region": "US",
    },
    "us_industrial_production": {
        "source": "fred",
        "id": "INDPRO",
        "label": "US industrial production",
        "unit": "index",
        "freq": "M",
        "region": "US",
    },
    "us_consumer_credit": {
        "source": "fred",
        "id": "TOTALSL",
        "label": "US consumer credit outstanding",
        "unit": "level",
        "freq": "M",
        "region": "US",
    },
    "us_auto_sales": {
        "source": "fred",
        "id": "TOTALSA",
        "label": "US light vehicle sales (SAAR)",
        "unit": "level",
        "freq": "M",
        "region": "US",
    },
    "us_auto_loan_rate": {
        "source": "fred",
        "id": "RIFLPBCIANM60NM",
        "label": "US 60M new-car loan rate",
        "unit": "pct",
        "freq": "M",
        "region": "US",
    },
}

# Statistical engine defaults (src/analysis/timeseries.py)
STATS_SIGNIFICANCE_LEVEL = 0.05  # p-value threshold for "statistically significant"
STATS_MIN_OBSERVATIONS = 12  # below this, correlations/regressions are not reported
STATS_MAX_LAG = 12  # lead/lag search window, in periods
STATS_ROLLING_WINDOW = 24  # rolling-correlation window, in periods
# Hodrick-Prescott smoothing by frequency (Ravn-Uhlig rule: 6.25 * periods_per_year^4)
HP_LAMBDA = {"D": 6.25 * 252**4, "W": 6.25 * 52**4, "M": 129_600.0, "Q": 1_600.0, "A": 6.25}
CYCLE_MIN_PERIOD_YEARS = 1.5  # business-cycle band for dominant-cycle search (Burns-Mitchell)
CYCLE_MAX_PERIOD_YEARS = 10.0
FORECAST_BACKTEST_FRACTION = 0.2  # last 20% of history held out to score forecast skill

# Rate-sensitivity (industry debt cost vs macro rate) analysis
RATE_SHOCK_BPS = [-100, -50, 50, 100, 200]  # scenario shocks to cost of debt
# Effective interest rate below this fraction of the benchmark rate is flagged
# as likely misclassified (e.g. captive-finance interest booked in COGS — GM, Ford)
RATE_SENSITIVITY_LOW_RATE_FLAG = 0.25
RATE_SENSITIVITY_MIN_COMPANIES = 3  # below this, no industry median/IQR findings are reported
# ...and above benchmark + this many pp as implausibly high (tiny debt base,
# one-off fees) — both directions are excluded from industry statistics.
RATE_SENSITIVITY_HIGH_SPREAD_FLAG_PP = 15.0
RATE_SENSITIVITY_MAX_COMPANIES = 15
RATE_SENSITIVITY_MIN_MARKET_CAP = 1_000_000_000  # screened universes only

# Relative valuation (stock PE vs peers vs industry vs sector vs market)
RELVAL_MAX_PEERS = 15
RELVAL_SECTOR_PEERS = 8  # size-matched same-sector reference set (never blended with industry peers)
RELVAL_MIN_LOOKUPS = 30  # candidate lookups per peer set, at least (sparse models need more)
RELVAL_MIN_MEANINGFUL_PEERS = 3  # below this, the sector set becomes the primary benchmark
RELVAL_MARKET_PROXY_SIZE = 30  # N largest US companies forming the market P/E aggregate
RELVAL_MARKET_PROXY_MIN_CAP = 100_000_000_000  # screener floor for the market proxy
RELVAL_MAX_MEANINGFUL_PE = 200.0  # P/E above this treated as not meaningful (near-zero earnings)
RELVAL_HISTORY_YEARS = 5

# Stock Analysis / Industry Comparison time series (src/analysis/indicators.py)
INDICATOR_HISTORY_YEARS = 10  # annual statements/ratios per company (Starter max; quarterly ratios are 402)
PRICE_HISTORY_YEARS = 5  # daily prices for return / risk metrics
PRICE_RISK_WINDOW_YEARS = 3  # volatility and beta window
MARKET_BENCHMARK_SYMBOL = "SPY"  # beta / relative-return benchmark (US-listed ETF, Starter OK)

# Industry comparison peer universe
INDUSTRY_COMP_MAX_PEERS = 10
INDUSTRY_COMP_MAX_PEERS_LIMIT = 25  # UI ceiling (each company costs ~7 FMP calls on first load)
INDUSTRY_COMP_SCREEN_LIMIT = 200  # screener rows fetched before cap/country filtering
INDUSTRY_COMP_MIN_CAP = 50_000_000  # screener floor: excludes shells / near-dormant listings
INDUSTRY_COMP_SIMILAR_SIZE_BAND = (0.2, 5.0)
# When fewer than INDUSTRY_COMP_MIN_FOR_AGGREGATE peers fall in the similar-size band, the
# band's LOWER bound is relaxed step by step (multiples of the subject's cap) before the peer
# scope widens. Mega-caps need this: 0.2x of AAPL's $5tn is $1tn, which excludes everyone.
PEER_SIZE_RELAX_STEPS = (0.2, 0.05, 0.01)
# Thin FMP industries -> adjacent FMP industries tried as "industry + related" BEFORE the
# whole sector (scopes are labelled and never blended). Every entry needs a reason.
RELATED_INDUSTRIES: dict[str, dict[str, Any]] = {
    # Kept symmetric: every relation is listed in both directions. Component suppliers ("Hardware,
    # Equipment & Parts": APH connectors, GLW glass, KEYS test gear) are deliberately NOT
    # related to device makers.
    "Consumer Electronics": {
        "related": ["Computer Hardware", "Communication Equipment"],
        "reason": "7 other listings, only SONY large (reports in JPY); device makers sit in adjacent industries",
    },
    "Computer Hardware": {
        "related": ["Consumer Electronics", "Communication Equipment"],
        "reason": "device and networking hardware are split across these FMP industries",
    },
    "Communication Equipment": {
        "related": ["Computer Hardware", "Consumer Electronics"],
        "reason": "only CSCO and MSI are large; networking hardware also sits in Computer Hardware",
    },
    "Insurance - Diversified": {
        "related": ["Insurance - Property & Casualty", "Insurance - Life"],
        "reason": "few US-listed diversified insurers (BRK fell back to banks and card networks)",
    },
    "Insurance - Property & Casualty": {
        "related": ["Insurance - Diversified"],
        "reason": "symmetric counterpart of Insurance - Diversified",
    },
    "Insurance - Life": {
        "related": ["Insurance - Diversified"],
        "reason": "symmetric counterpart of Insurance - Diversified",
    },
}
INDUSTRY_COMP_MIN_FOR_AGGREGATE = 3  # fewer companies than this -> aggregate not reported
# Market-cap buckets (USD, lower bound inclusive / upper exclusive; None = unbounded)
CAP_SIZE_BUCKETS: dict[str, tuple[float, float | None]] = {
    "Mega (≥ $200B)": (200e9, None),
    "Large ($10B–200B)": (10e9, 200e9),
    "Mid ($2B–10B)": (2e9, 10e9),
    "Small ($300M–2B)": (300e6, 2e9),
    "Micro (< $300M)": (0.0, 300e6),
}
# P/E peers whose latest-FY EPS changed outside this range are skipped as cyclical-peak or
# one-off earnings (SNDK +752%, WDC +374% made AAPL look 46% overvalued vs hardware peers).
PEER_PE_EPS_SWING = (-0.67, 2.0)
# A Σcap÷Σdenominator aggregate dominated by one peer (> this share of Σcap) that also sits
# more than PEER_AGG_MEDIAN_GAP from the peer median measures that one company, not the group
# (DELL vs ANET 52%: 30.9x vs median 23.6x; COF vs AXP 74%: 3.5x vs 1.9x) -> not credible.
PEER_DOMINANCE_SHARE = 0.50
PEER_AGG_MEDIAN_GAP = 0.25
# Concentration alone: a "peer set" that is effectively one company (effective peer count
# 1/Σw² below this, or one peer above PEER_MAX_SINGLE_SHARE of Σcap) is a one-company
# comparison even when it agrees with the median (GOOGL vs META 97%, META vs GOOG 99%).
PEER_MIN_EFFECTIVE_N = 2.0
PEER_MAX_SINGLE_SHARE = 0.75

# Industry-specific indicator profiles. A company's FMP industry is matched
# (case-insensitive substring) against "industries" first, then its sector
# against "sectors"; first match wins, else "default". "defaults" are the
# indicators shown initially per group; "not_meaningful" are flagged if the
# user selects them (e.g. gross margin / current ratio / EV/EBITDA for banks,
# whose balance sheets are funded by deposits). Keys are indicator keys from
# src/analysis/indicators.py.
INDUSTRY_INDICATOR_PROFILES: dict[str, dict[str, list[str] | str]] = {
    "bank": {
        "label": "Banks",
        # Not "Financial - Credit Services": FMP files payment networks (V, MA) there,
        # and bank defaults would mark their gross/FCF margins "not meaningful".
        "industries": ["Banks"],
        "sectors": [],
        "return": [
            "roe",
            "roa",
            "netInterestShare",
            "efficiencyRatio",
            "epsGrowth",
            "netRevenueGrowth",
        ],
        "risk": ["equityToAssets", "financialLeverage", "incomeQuality", "volatility", "maxDrawdown", "beta"],
        "valuation": ["pe", "pb", "pTangibleBook", "dividendYield", "payoutRatio"],
        "not_meaningful": [
            # gross revenue includes interest income; margins aren't comparable without COGS
            "netMargin",
            "revenueGrowth",
            "grossMargin",
            "operatingMargin",
            "ebitdaMargin",
            "currentRatio",
            "quickRatio",
            "cashRatio",
            "evToEbitda",
            "evToSales",
            "netDebtToEbitda",
            "gearing",
            "interestCoverage",
            "cashConversionCycle",
            "inventoryTurnover",
            "fcfMargin",
            "pFcf",
        ],
    },
    "insurance": {
        "label": "Insurance",
        "industries": ["Insurance"],
        "sectors": [],
        "return": ["roe", "roa", "netMargin", "revenueGrowth", "epsGrowth"],
        "risk": ["equityToAssets", "financialLeverage", "debtToCapital", "volatility", "maxDrawdown", "beta"],
        "valuation": ["pe", "pb", "pTangibleBook", "dividendYield", "payoutRatio"],
        "not_meaningful": [
            "grossMargin",
            "currentRatio",
            "quickRatio",
            "cashRatio",
            "evToEbitda",
            "netDebtToEbitda",
            "cashConversionCycle",
            "inventoryTurnover",
        ],
    },
    "reit": {
        "label": "REITs",
        "industries": ["REIT"],
        "sectors": ["Real Estate"],
        "return": ["ffoMargin", "revenueGrowth", "roe", "operatingMargin", "dividendYield"],
        "risk": ["netDebtToEbitda", "debtToAssets", "interestCoverage", "gearing", "volatility", "beta"],
        "valuation": ["pFfo", "pb", "evToEbitda", "dividendYield", "payoutRatio"],
        "not_meaningful": ["grossMargin", "inventoryTurnover", "cashConversionCycle", "epsGrowth"],
    },
    "software": {
        "label": "Software & IT services",
        "industries": ["Software", "Information Technology Services", "Internet Content"],
        "sectors": [],
        "return": ["revenueGrowth", "grossMargin", "operatingMargin", "fcfMargin", "ruleOf40", "roic"],
        "risk": ["sbcToRevenue", "gearing", "currentRatio", "intangiblesToAssets", "volatility", "beta"],
        "valuation": ["evToSales", "pe", "pFcf", "fcfYield"],
        "not_meaningful": ["inventoryTurnover"],
    },
    "semis": {
        "label": "Semiconductors",
        "industries": ["Semiconductors"],
        "sectors": [],
        "return": ["revenueGrowth", "grossMargin", "operatingMargin", "fcfMargin", "roic", "rdIntensity"],
        "risk": ["gearing", "currentRatio", "inventoryTurnover", "capexToRevenue", "volatility", "beta"],
        "valuation": ["pe", "evToEbitda", "evToSales", "pFcf"],
        "not_meaningful": [],
    },
    "retail": {
        "label": "Retail",
        "industries": [
            "Retail",
            "Department Stores",
            "Discount Stores",
            "Grocery Stores",
            "Home Improvement",
        ],
        "sectors": [],
        "return": ["revenueGrowth", "grossMargin", "operatingMargin", "roic", "assetTurnover"],
        "risk": [
            "inventoryTurnover",
            "cashConversionCycle",
            "gearing",
            "netDebtToEbitda",
            "currentRatio",
            "volatility",
        ],
        "valuation": ["pe", "evToEbitda", "evToSales", "dividendYield"],
        "not_meaningful": [],
    },
    "energy": {
        "label": "Energy",
        "industries": [],
        "sectors": ["Energy"],
        "return": ["roce", "operatingMargin", "fcfMargin", "revenueGrowth", "dividendYield"],
        "risk": ["netDebtToEbitda", "gearing", "interestCoverage", "capexToRevenue", "volatility", "beta"],
        "valuation": ["evToEbitda", "pe", "pFcf", "fcfYield", "dividendYield"],
        "not_meaningful": [],
    },
    "utilities": {
        "label": "Utilities",
        "industries": [],
        "sectors": ["Utilities"],
        "return": ["roe", "operatingMargin", "revenueGrowth", "epsGrowth", "dividendYield"],
        "risk": ["netDebtToEbitda", "gearing", "interestCoverage", "debtToCapital", "capexToRevenue", "beta"],
        "valuation": ["pe", "evToEbitda", "pb", "dividendYield", "payoutRatio"],
        "not_meaningful": ["inventoryTurnover", "cashConversionCycle"],
    },
    "default": {
        "label": "General (non-financial)",
        "industries": [],
        "sectors": [],
        "return": [
            "revenueGrowth",
            "epsGrowth",
            "grossMargin",
            "operatingMargin",
            "netMargin",
            "roe",
            "roic",
        ],
        "risk": [
            "gearing",
            "netDebtToEbitda",
            "interestCoverage",
            "currentRatio",
            "volatility",
            "maxDrawdown",
            "beta",
        ],
        "valuation": ["pe", "evToEbitda", "pb", "pFcf", "dividendYield"],
        "not_meaningful": [],
    },
}
