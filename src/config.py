# src/config.py

import os
from pathlib import Path

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
DCF_CAPTIVE_DEBT_WEIGHT = 0.50
# Intrinsic value / price outside this band is reported but never blended into a target.
DCF_SANITY_BAND = (0.2, 5.0)
# Terminal value above this share of EV is flagged (the answer is mostly the TV assumption).
DCF_TV_SHARE_FLAG = 0.75

# Business-model classification (src/analysis/business_model.py). Matched as substrings
# of the FMP industry name, first match wins (order matters: "REIT - Mortgage" is a lender).
BUSINESS_MODEL_INDUSTRY_KEYWORDS: list[tuple[str, str]] = [
    ("REIT - Mortgage", "lender"),
    ("REIT", "reit"),
    ("Banks", "bank"),
    ("Capital Markets", "bank"),
    ("Investment - Banking", "bank"),
    ("Financial - Mortgages", "lender"),
    ("Insurance", "insurer"),
    ("Asset Management", "asset_manager"),
]
# Financial Services companies not caught above are lenders when gross interest income
# is at least this share of revenue. Calibrated on the local store (FY2021-25): payment
# networks/processors <= 2.1% (V, MA 0; PYPL, GPN ~2%), lenders >= 20% (AXP 20-32%,
# SOFI 33-76%, COF ~80%, ALLY 81-111%, SYF 109-136%). FMP sometimes reports 0 for an
# unmapped field (AFRM 2022-25), so the rule can miss a lender, never flag a payments firm.
LENDER_INTEREST_INCOME_SHARE = 0.10

# Peer-multiple valuation (relative_valuation.peer_valuation): method by business model.
PEER_METHOD_BY_KIND = {
    "standard": "pe",
    "asset_manager": "pe",
    "bank": "pb",
    "lender": "pb",
    "insurer": "pb",
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
INDUSTRY_COMP_SIMILAR_SIZE_BAND = (0.2, 5.0)  # "similar size" = 0.2x-5x of the subject's market cap
INDUSTRY_COMP_MIN_FOR_AGGREGATE = 3  # fewer companies than this -> aggregate not reported
# Market-cap buckets (USD, lower bound inclusive / upper exclusive; None = unbounded)
CAP_SIZE_BUCKETS: dict[str, tuple[float, float | None]] = {
    "Mega (≥ $200B)": (200e9, None),
    "Large ($10B–200B)": (10e9, 200e9),
    "Mid ($2B–10B)": (2e9, 10e9),
    "Small ($300M–2B)": (300e6, 2e9),
    "Micro (< $300M)": (0.0, 300e6),
}

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
            "netMargin",
            "netInterestShare",
            "efficiencyRatio",
            "epsGrowth",
            "revenueGrowth",
        ],
        "risk": ["equityToAssets", "financialLeverage", "incomeQuality", "volatility", "maxDrawdown", "beta"],
        "valuation": ["pe", "pb", "pTangibleBook", "dividendYield", "payoutRatio"],
        "not_meaningful": [
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
