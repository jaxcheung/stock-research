# Stock Research & Asset Management

A personal research tool for equity analysis, quantitative macro work and
portfolio tracking. It pulls market and fundamental data from Financial
Modeling Prep (FMP), macro data from FRED and the World Bank, stores history
locally as Parquet (queried with DuckDB), and presents everything in a
Streamlit dashboard with downloadable HTML reports.

Every result comes with its methodology, statistical significance and
data-quality caveats. The tool is built to flag its own weak spots, not just
produce plausible-looking numbers.

> For personal research only. Not investment advice.

---

## Contents

1. [Quick start](#1-quick-start)
2. [Configuration](#2-configuration)
3. [The dashboard](#3-the-dashboard)
4. [Downloading data in bulk (CLI)](#4-downloading-data-in-bulk-cli)
5. [Using the modules from Python](#5-using-the-modules-from-python)
6. [Series specs: adding your own data](#6-series-specs-adding-your-own-data)
7. [Where data is stored](#7-where-data-is-stored)
8. [Methodology in brief](#8-methodology-in-brief)
9. [FMP Starter-tier limits](#9-fmp-starter-tier-limits)
10. [Troubleshooting](#10-troubleshooting)
11. [Development](#11-development)

---

## 1. Quick start

Requires **Python 3.10**.

```bash
cd ~/projects/stock-research
pip install -r requirements-dev.txt        # runtime deps + ruff/mypy
cp .env.example .env                       # then add your FMP key (see §2)
streamlit run src/dashboard/app.py         # opens http://localhost:8501
```

Optional: load 10 years of income statements for the Fundamentals Explorer
(about 3–4 minutes):

```bash
python -m src.data.sync
```

## 2. Configuration

Secrets go in `.env` (gitignored, never commit it). The template is `.env.example`.

| Variable | Required | Purpose |
|---|---|---|
| `FMP_API_KEY` | **Yes** | Financial Modeling Prep key (Starter tier is enough). |
| `FRED_API_KEY` | No | Official FRED API. Without it, FRED's keyless public CSV is used (same series, full history). |
| `LOG_LEVEL` | No | `INFO` (default), `DEBUG`, `WARNING`… |
| `CACHE_ENABLED` | No | `0` disables the JSON response cache (debugging only). |

All other tunable settings live in **`src/config.py`**: DCF assumptions, cache
TTLs, statistical thresholds, rate-shock scenarios, peer-group sizes, the
macro series catalog and the FMP rate limit. Change behaviour there, not in
the modules.

## 3. The dashboard

```bash
streamlit run src/dashboard/app.py
```

Pages are grouped in the sidebar. Set a **default ticker** on Home; it carries
over to every page. Each analysis page follows the same layout: inputs →
**KPI tiles** → **key findings** → tabbed charts and tables → caveats →
methodology and sources → **Download full report (HTML)**.

### Research

| Page | What it does |
|---|---|
| **Stock Analysis** | Up to 10 fiscal years of return, risk (incl. gearing) and valuation indicators, statements in $, price return/risk. Industry-specific indicator defaults; pick/hide any indicator. |
| **Industry Comparison** | Return and risk compared separately vs a filtered peer group (scope: industry / sector / FMP peers / custom; cap-size buckets or similar size; US-only), time series + snapshot, industry aggregates (median, Σ÷Σ e.g. industry P/E). |
| **DCF Valuation** | Two-stage FCF model, CAPM WACC, live 10Y Treasury as risk-free rate. |
| **Research Report** | Full equity report: scorecard, DCF + comps price target, risks. The quality rating and the valuation verdict are reported separately. |

### Quant Lab

**Macro Lab: multi-variable statistics**
1. Pick a **preset** (e.g. *China rates & inflation*) or choose series from the catalog.
2. Optionally add any series by spec, e.g. `fred:DGS2, price:TSLA` (see §6).
3. Choose the **target** variable, the frequency (*Auto* = coarsest, never interpolates)
   and a **stationary transform** (e.g. *Year-over-year change* for rates).
4. **Run analysis.** The tabs are:
   - *Overview*: indexed or z-scored chart, small multiples, descriptive statistics and stationarity tests (ADF).
   - *Correlation*: heatmaps on levels **and** on the transform, with p-values, plus rolling correlation.
   - *Lead / lag*: which series moves first, and by how many periods.
   - *Regression*: target on all drivers, with Newey-West (HAC) standard errors.
   - *Cycles*: trend/cycle split, dominant cycle length, current phase (expansion, slowdown, contraction, recovery).
   - *Forecast*: cycle-aware projection with a 95% band, back-tested against a naive forecast.

**Rate Sensitivity: industry cost of debt vs a macro rate**
- *Screen an industry*: e.g. Industry `Auto - Manufacturers`, country `China`,
  benchmark `China 3M interbank rate`.
- *Custom tickers*: e.g. `GM, F, TSLA, STLA` against `US Fed funds rate`.
- Output: effective borrowing rate per company and for the industry median,
  credit spread, interest coverage, pass-through β (panel regression) and a
  rate-shock heatmap from −100 to +200 bp.
- Implausible data is excluded and explained. For example, GM and Ford's
  finance-arm interest is booked outside "interest expense".

**Relative Valuation: P/E in context**
- Enter a ticker. The page compares its TTM P/E against **industry peers**,
  **size-matched sector peers** (never blended with industry peers), FMP
  industry and sector P/E, a **US large-cap market proxy**, and its **own
  history** (P/E built from point-in-time EPS, ±1σ bands).
- The verdict is *Premium / In line / Discount*: a relative-valuation reading,
  not a Buy/Sell rating.
- The first run builds the 30-company market proxy (about 1 minute); later runs are cached.

**Fundamentals Explorer: 10-year panel (Parquet + DuckDB)**
- *Sector trends*: sector × year heatmap and trend lines for 11 ratios
  (margins, R&D and SG&A intensity, interest burden, coverage, tax rate, growth).
- *Leaderboard*: revenue and EPS CAGR, margin level, trend and volatility; a
  growth-vs-profitability scatter. Companies growing from a tiny base are flagged.
- *Company*: revenue, net income and margin history for one company.
- *SQL workbench*: read-only DuckDB queries over the store (examples included).
- *Data store*: what's stored, plus a form to download or refresh data with a progress bar.

### Portfolio

| Page | What it does |
|---|---|
| **Portfolio** | Record BUY/SELL transactions; FIFO realised PnL and live unrealised PnL. |
| **Knowledge Log** | Save and search investing quotes and lessons; random "quote of the moment". |

## 4. Downloading data in bulk (CLI)

`src/data/sync.py` downloads FMP statements into the local Parquet store.

```bash
# Default: US-listed companies >= $10B, 10 years of annual income statements
python -m src.data.sync

# Broader universe and all three statements (~4,800 calls ≈ 20 min)
python -m src.data.sync --universe us_mid --min-cap 2e9 \
    --kinds income_statement balance_sheet cash_flow

# Explicit tickers (e.g. Chinese EV makers) into their own universe
python -m src.data.sync --universe china_autos --symbols NIO XPEV LI BYDDY

# Quarterly statements (40 quarters for 10 years)
python -m src.data.sync --period quarter --years 10

# Force a refresh even if files are recent
python -m src.data.sync --max-age-hours 0
```

| Flag | Default | Meaning |
|---|---|---|
| `--universe` | `us_large` | Name for the company-list snapshot. |
| `--min-cap` | `10e9` | Screener market-cap floor in USD. |
| `--country` | `US` | Screener country (ISO2); `''` for any. |
| `--symbols` | none | Explicit tickers instead of a screen. |
| `--kinds` | `income_statement` | Any of `income_statement balance_sheet cash_flow`. |
| `--period` | `annual` | `annual` or `quarter`. |
| `--years` | `10` | Years of history (Starter tier returns up to 10 annual). |
| `--max-age-hours` | `168` | Skip companies refreshed within this window. |

Behaviour:
- **Safe to re-run or interrupt.** Each company is saved independently and
  files refreshed within `--max-age-hours` are skipped, so a re-run picks up
  where it stopped.
- **One bad ticker never stops the batch.** Failures are listed in the summary.
- **Rate-limited** to `FMP_MAX_CALLS_PER_MINUTE` (250, under Starter's ~300).
- **History accumulates.** New downloads merge into stored history (restatements
  replace old values) and never overwrite rows the API no longer returns.
- **Cleaned universe.** Exchange-listed bonds and preferred shares are filtered
  out, and duplicate share classes (GOOG/GOOGL) are collapsed.

## 5. Using the modules from Python

Run from the project root so `src` is importable.

```python
# Multi-variable macro analysis
from src.analysis.macro_lab import run_macro_lab
rep = run_macro_lab(
    ["cn_interbank_3m", "cn_cpi_yoy", "us_fed_funds", "usd_cny"],
    freq="M", transform="yoy_diff", start="2010-01-01", target="cn_interbank_3m",
)
for f in rep.findings:
    print(f.text)
rep.stats              # descriptive statistics + ADF
rep.corr_transformed   # {"r", "p", "n"} DataFrames
rep.cycles["China 3M interbank rate"]["phase"]

# Industry rate sensitivity
from src.analysis.rate_sensitivity import run_rate_sensitivity
rs = run_rate_sensitivity("cn_interbank_3m", industry="Auto - Manufacturers", country="CN")
rs.industry, rs.companies, rs.scenarios

# Relative valuation
from src.analysis.relative_valuation import run_relative_valuation
rv = run_relative_valuation("AAPL", max_peers=10)
rv.verdict, rv.verdict_detail, rv.benchmarks

# 10-year fundamentals panel (needs a sync first)
from src.analysis.fundamentals_panel import analyze_panel, sector_trend, growth_leaderboard
panel = analyze_panel()
medians, counts = sector_trend(panel.panel, "operatingMargin", panel.complete_years)
leaders = growth_leaderboard(panel.panel, 2017, 2025)

# Ad-hoc DuckDB over the store
from src.data.store import StoreDB
db = StoreDB()
db.query("SELECT symbol, max(revenue) FROM income_annual GROUP BY 1 ORDER BY 2 DESC LIMIT 10")

# Export any analysis as a self-contained HTML report
from src.reporting import builders
open("macro.html", "w").write(builders.macro_lab_html(rep))
# also: rate_sensitivity_html(rs), relative_valuation_html(rv),
#       panel_html(panel, "operatingMargin", ["Technology", "Energy"])
```

Statistical building blocks are in `src/analysis/timeseries.py` (pure
functions, no I/O): `align`, `transform`, `adf_test`, `descriptive_stats`,
`correlation_matrix`, `lead_lag`, `ols_hac`, `ols_clustered`, `hp_filter`,
`cycle_analysis`, `forecast`.

## 6. Series specs: adding your own data

Anywhere a series is accepted (Macro Lab, `load_series`, `run_macro_lab`),
you can pass a **catalog key** or a **spec**:

| Spec | Example | Source |
|---|---|---|
| catalog key | `cn_interbank_3m` | `MACRO_SERIES_CATALOG` in `src/config.py` |
| `fred:<ID>` | `fred:DGS2` | Any FRED series ID |
| `worldbank:<ISO3>:<indicator>` | `worldbank:IND:FR.INR.LEND` | World Bank WDI (annual) |
| `price:<ticker>` | `price:TSLA` | FMP daily close |
| `sectorpe:<sector>` | `sectorpe:Technology` | FMP historical sector P/E |
| `industrype:<industry>` | `industrype:Auto - Manufacturers` | FMP historical industry P/E |

To make a series permanent, add an entry to `MACRO_SERIES_CATALOG`. The
`unit` field matters: rates should be `"pct"` (percentage points), so cycles
are measured in pp, not "% from trend".

## 7. Where data is stored

Everything lives under `data/`, which is gitignored.

| Path | Format | Role |
|---|---|---|
| `data/store/` | Parquet | **System of record** for statements, universes and series. Accumulates history; don't delete casually. |
| `data/cache/` | JSON | Disposable API response cache with TTLs. Safe to delete. |
| `data/portfolio.db` | SQLite | Your transactions (personal financial data). |
| `data/knowledge.db` | SQLite | Quotes and lessons. |
| `log/` | text | Rotating logs, one file per module. |

Store layout:
```
data/store/fmp/statements/<income_statement|balance_sheet|cash_flow>/<annual|quarter>/<SYMBOL>.parquet
data/store/fmp/universes/<name>.parquet
data/store/series/<fred|worldbank|price|...>/<ID>.parquet
```
DuckDB tables: `income_annual`, `income_quarter`, `balance_annual`, `cashflow_annual`, `companies`, `series`.

If FRED, the World Bank or FMP is unreachable, the series loader falls back to
the stored Parquet copy and adds a caveat with the date of the last observation.

## 8. Methodology in brief

- **Currency consistency.** P/E and other price-to-financials ratios are nulled
  or refused when a company quotes in one currency and reports in another
  (e.g. Sony ADR vs JPY financials). Cross-company comparisons use ratios only,
  never sums of amounts in different currencies.
- **No spurious correlations.** Every series is tested for stationarity (ADF).
  Inference uses stationary transforms, and level correlations that vanish
  after transformation are flagged.
- **Honest standard errors.** HAC (Newey-West) for time series;
  company-clustered errors for panels.
- **Forecasts are scored.** Each forecast is back-tested out of sample against
  a naive forecast. If it doesn't beat naive, it is labelled illustrative.
- **Correct aggregation.** Group P/E = Σ market cap ÷ Σ earnings, never an
  average of P/Es. Cross-sections use medians, and cells with fewer than
  5 companies are suppressed.
- **No look-ahead.** Historical P/E uses each quarter's EPS only from its filing date.
- **Quality rating ≠ valuation.** The Buy/Hold/Sell quality score and the
  valuation verdict are always reported separately.
- **Known biases are disclosed.** For example, survivorship bias in the
  large-cap panel, captive-finance interest, and acquisition-driven growth.

Each report's *Methodology* section has the full details for that analysis.

## 9. FMP Starter-tier limits

Verified on a Starter key (2026-11-05):

| Available | Premium only (402) |
|---|---|
| Statements (10y annual / 40 quarters), ratios and key-metrics incl. TTM, daily prices, sector and industry P/E snapshots and history, company screener, treasury rates, industry/sector/country lists | `batch-quote`, `sp500-constituent`, foreign-exchange tickers (e.g. `002241.SZ`) |

Foreign tickers are filtered out automatically. US-listed ADRs (NIO, TM, BYDDY…)
work, subject to the currency rules above.

## 10. Troubleshooting

| Symptom | Fix |
|---|---|
| `FMP_API_KEY is not set` | Add it to `.env` in the project root. |
| `No income statements in the local store` | Run `python -m src.data.sync` or use *Fundamentals Explorer → Data store*. |
| `… quotes in a different currency than it reports` | Expected for foreign ADRs in valuation analyses (FX conversion is not implemented). |
| A peer or company is missing from results | Check the page's *Caveats*: it was probably excluded for currency mismatch, a negative or extreme P/E, or a data-quality flag. |
| Numbers look stale | Delete `data/cache/` (safe), or re-sync with `--max-age-hours 0`. |
| `402` errors in logs | That endpoint or ticker needs FMP Premium (see §9). Batch operations skip it and continue. |
| Slow first run | Relative Valuation's market proxy and bulk syncs make many API calls once; later runs are cached. |

Logs are in `log/<module>.log`.

## 11. Development

```bash
ruff check src/ && ruff format --check src/   # lint + format
mypy                                           # type check (config in pyproject.toml)
```

- There's no automated test suite by design (personal, local project). Verify
  changes by running the affected function on real symbols (`AAPL`, a
  currency-mismatch ADR like `SONY`, and a partial-data case), and drive changed
  dashboard pages headlessly with `streamlit.testing.v1.AppTest`.
- Project rules (where settings live, logging, currency checks, statistical
  standards, chart conventions) are in **`CLAUDE.md`**. Read it before changing code.
- Dependencies are pinned. Upgrade one at a time and re-verify.
