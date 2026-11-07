# Stock Research & Asset Management App

## Introduction
A personal stock research and portfolio management tool, aiming for
professional/CFA-grade analytical rigor rather than toy calculations. It pulls
market and fundamental data from Financial Modeling Prep (FMP) and macro data
from FRED, produces equity research reports (DCF + peer-multiple valuation,
CFA-style composite scoring, identified risks, rating & price target), tracks
the user's own transactions and PnL, and keeps a running log of investing
quotes/lessons (books, Warren Buffett, etc.).

## Goal
Built incrementally, starting cheap:
1. Working FMP client and fundamentals module on the **Starter** API tier. (done)
2. Portfolio/PnL tracking and the quotes/knowledge log. (done)
3. Production-grade analysis: macro data (FRED), CFA-style investment scoring
   that actively flags opportunities, and a full equity research report —
   rigorous enough to catch its own mistakes (see the currency-mismatch and
   rating/valuation-conflation fixes in Rules) rather than just producing
   plausible-looking numbers. (done — see Milestones)
4. Upgrade the FMP plan only once a specific feature is blocked by a
   Starter-tier limit (e.g. full historical statements) — not preemptively.

## Rules
- Never commit API keys or `.env` files. Load secrets via environment variables
  (`FMP_API_KEY`, etc.), never hardcode them. `.env.example` stays blank/template only.
- All modules log through `src/utils/logging.py` (`get_logger`), not bare `print`.
- All env-driven and default settings live in `src/config.py` — don't read
  `os.environ` or hardcode tunable constants (API base URLs, DCF assumptions,
  peer-comparison limits, etc.) directly in other modules.
- Keep data-fetching (FMP/news clients) separate from analysis logic, and both
  separate from portfolio/transaction logic.
- Assume FMP **Starter** tier endpoint limits unless told otherwise — flag it
  explicitly if an endpoint needs Premium. Code defensively around partial data
  (e.g. foreign-exchange peer tickers 402'ing, missing interest expense) rather
  than letting one bad symbol kill a batch operation.
- FMP's API migrated to the `/stable` base with query-param symbols (`?symbol=`)
  — the legacy `/api/v3/...` path-based endpoints are deprecated and 403 for
  new subscriptions.
- **Storage tiers:** `data/cache/` (JSON) is a disposable request cache;
  `data/store/` (Parquet) is the system of record for market/fundamental
  history — never delete it casually, it accumulates history the API's
  look-back window no longer returns. SQLite stays for portfolio/knowledge
  records. FMP calls are throttled to `FMP_MAX_CALLS_PER_MINUTE`.
- Transaction/PnL data is user financial data — treat it as sensitive, store
  locally, never log full transaction contents at INFO level with secrets.
  Stored in SQLite (`data/portfolio.db`, gitignored), not CSV/JSON.
- Realized PnL uses **FIFO** cost-basis matching (oldest open lot sold first) —
  don't switch to average-cost without an explicit decision to do so. A SELL is
  validated against the (date, id)-ordered timeline, not the all-time net: a
  backdated SELL must not exceed shares held on its date or starve a later SELL.
- FMP requests retry with exponential backoff on transient failures (network
  errors, 429/500/502/503/504) and fail fast on non-retryable errors
  (402/403/404) — see `FMP_MAX_RETRIES` etc. in `src/config.py`. Don't add a
  second retry layer on top; extend the one in `fmp_client._get`.
- Validate inputs at module boundaries with clear `ValueError`s (symbol
  normalization, `limit`/`max_peers` >= 1, DCF assumption ranges) rather than
  letting bad input surface as a confusing downstream error.
- Code must pass `ruff check` and `mypy` clean (config in `pyproject.toml`)
  before considering a change done. No test suite / CI by deliberate choice —
  this is a personal, local-only project; re-evaluate if that scope changes.
- **Currency consistency is mandatory, not optional.** Foreign ADRs (Sony,
  Panasonic, Samsung, Xiaomi, ...) quote in USD but report financials in their
  home currency — mixing them silently produces nonsense (a real DCF run
  produced a "1813% upside" before this was caught). Before computing any
  ratio or valuation that divides a price/market-cap figure by a per-share
  financial-statement figure, compare `profile["currency"]` against the
  statement's `reportedCurrency`. On mismatch: null the affected fields (don't
  drop the whole record — profitability/growth/leverage ratios computed
  entirely within the reporting currency are still valid) and surface it
  explicitly (a `currencyMismatch` flag, a warning log, and a note in any
  rendered report). `dcf.py` treats a mismatch as a hard `ValueError` since the
  entire DCF output — not just part of it — is corrupted without FX conversion
  (not implemented). Apply this same check to any new analysis that mixes
  price/market-cap with statement-derived per-share data.
- **Value a business with methods that fit its model.** `business_model.classify`
  (pure, no I/O) labels every company `standard` / `bank` / `lender` / `broker_dealer` /
  `insurer` (life | pc) / `asset_manager` / `reit`, plus a `captive_finance` flag. Order:
  config overrides → strong keywords (REIT, Banks, Insurance) → gross interest income ÷
  revenue (>= 10% lender; in Capital Markets/Asset Management 10–80% broker-dealer, >= 80%
  lender subtype `bdc` only if equity >= 30% of assets — the 150% asset-coverage rule — so
  annuity insurers filed as Asset Management (CRBG, equity 3%) aren't BDCs; "REIT - Mortgage"
  is lender subtype `mreit`) → unmapped-interest test (interest < 0.5% but receivables >= 40% of assets →
  lender) → weak keywords → standard. Calibration and every override's reason live in
  `config.py`; overrides are always disclosed in reports. Captive finance = debt-funded
  receivables ((receivables − payables) >= 40% of assets AND debt >= 0.6× receivables), or
  cost of debt far below risk-free with debt > 50% of capital, or interest income >= 5% of
  revenue with debt > 50% of capital. Receivables alone flagged freight brokers, staffing
  firms and distributors (trade credit, funded by payables); a low cost of debt alone flags
  GOOGL/XOM (legacy coupons). Overrides may also set `peer_industries`, and
  `KIND_PEER_INDUSTRIES` widens the industry peer screen for models FMP spreads across
  industries (broker-dealers, BDCs).
  Consequences:
  - `dcf.py` raises `ValueError` (→ "DCF unavailable: <reason>") for every non-`standard`
    model and for captive finance, cost of debt > `DCF_MAX_DERIVED_COST_OF_DEBT`,
    non-positive base FCF, non-positive equity value, and a WACC below the risk-free rate
    (raw low betas: TMUS 0.32 gave WACC 4.67% and +282%). A derived cost of debt more than
    `DCF_CAPTIVE_COD_GAP` below risk-free is replaced by risk-free + `DCF_DEBT_SPREAD`
    (marginal, not historical, cost — disclosed). Derived FCF growth is clamped. Output
    carries `terminalValueShare` and `sanityBandBreach` (outside `DCF_SANITY_BAND` × price).
  - Peer valuation (`relative_valuation.peer_valuation`) uses P/E (standard, asset
    managers), ROE-adjusted P/B (bank / lender / broker-dealer / insurer: warranted P/B =
    peer P/B × (ROE_s − g)/(ROE_p − g), ROE = TTM earnings ÷ book for subject AND peers;
    loss-making peers excluded; subject and peer ROE must each exceed g by
    `PEER_ROE_MIN_SPREAD`; subject ROE within the peer range ± `PEER_ROE_RANGE_TOLERANCE`;
    else "unreliable"). BDCs and mortgage REITs (`PLAIN_PB_SUBTYPES`) use plain P/B ≈
    price/NAV against same-subtype peers — lender subtypes must match (ARCC was −48% against
    card lenders). or P/FFO (REITs). Peers share the subject's
    business model and are operating equities; one scope at a time; Σ÷Σ + median; >=
    `PEER_TARGET_MIN_PEERS` usable peers (usability — e.g. no loss-makers for P/B — is
    checked DURING selection, so the search continues until enough are found) or no
    implied price.
  - **`valuation_summary.summarize_valuation` is the single rule** for which methods count
    and what the valuation verdict is; the report's price target AND the scorecard's
    valuation signal both use it. Credible methods that disagree (one Attractive and one
    Rich, or more than 30 points apart) → "Indeterminate — methods disagree", whichever is
    in band; neighbouring verdicts close together (−9% vs −11%) agree. Two agreeing methods with one
    within ±`TARGET_MAX_SINGLE_METHOD_GAP` → point target, full score. Two agreeing, both
    beyond → "low precision" range. One method within the band → "(single method)". In both
    of those, the score is capped to `VALUATION_SCORE_CAP`. One method beyond the band →
    "Indicative only", unscored. Strong valuation scores need corroboration.
  - Scorecard: banks/lenders/broker-dealers are scored on equity ÷ assets; insurers'
    financial health is not scored (separate-account assets make equity/assets meaningless;
    RBC/combined ratio unavailable); negative equity makes D/E and ROE "n/m"; captive
    finance leaves financial health unscored (consolidated current ratio, D/E and coverage all
    mix in the finance arm). Each adds an explicit risk line.
- **Screener defaults:** `FMPClient.screen_companies` excludes ETFs and funds
  (`isEtf`/`isFund=false`) because FMP's response flags are unreliable — "Asset Management"
  returned 100 funds and no BDCs. Peer screens also run a size-banded pass per industry,
  since the screener returns at most 100 rows, largest first.
- **Thin peer sets are escalated, never silently replaced.** Industry Comparison tries:
  own FMP industry → the similar-size band's lower bound relaxed (`PEER_SIZE_RELAX_STEPS`,
  floor shown in dollars) → own + `RELATED_INDUSTRIES` (symmetric config map, a reason per
  entry) → whole sector. A step succeeds only with enough SAME-COUNTRY peers (proxy for a
  comparable reporting currency). The subject's own share classes are never peers (BRK-A for
  BRK-B). The size band is applied SERVER-side; the screener returns at most 200 rows largest
  first, and truncation is shown. A **peer funnel** explains each attempt, and labels say
  "Related industries only" when no own-industry peer survives.
  - In the research report, related industries are a FALLBACK only, used when the own
    industry yields fewer than `PEER_TARGET_MIN_PEERS` usable peers (applying them always
    flipped DELL −16% → +19%).
  - A related-only peer set is marked unreliable (indicative).
  - P/E peers with latest-FY EPS change outside `PEER_PE_EPS_SWING` are skipped as
    cyclical-peak or one-off earnings (SNDK +752%).
  - An aggregate is unreliable when the peer set is effectively one company (effective peer
    count 1/Σw² < `PEER_MIN_EFFECTIVE_N`, or one peer > `PEER_MAX_SINGLE_SHARE` of Σcap —
    GOOGL vs META 97%), or when one peer is > `PEER_DOMINANCE_SHARE` AND the aggregate is more
    than `PEER_AGG_MEDIAN_GAP` from the median (DELL vs ANET). Excluded methods that point the
    other way are disclosed next to the verdict, but don't veto it.
  - If the whole-sector fallback finds fewer comparable peers than the best industry attempt,
    the industry attempt is kept and the result is labelled thin (GOOGL).
- **Data quality (`src/analysis/data_quality.py`).** Stub statement rows (latest period with
  equity and liabilities zero, or assets < `STUB_STATEMENT_ASSET_RATIO` of the prior period)
  are never read as real: analyses fall back to the prior period — balance sheet AND the
  ratios/key metrics of that period, and FY book for peer P/B when TTM P/B is a stub — and
  surface a "Data quality" note (RJF FY2026 read as "$0 equity" before). `check_statements(income,
  balance, cashflow)` → `DataQualityReport` of warn/info issues: balance identity, FCF =
  CFO − capex, income vs cash-flow net income tie-out (accepted when cash-flow NI matches
  attributable, continuing, or continuing + discontinued income; minority interests and
  discontinued operations are info notes; a match to PRE-TAX income is a mis-mapping
  warning (TSM), tested only when tax is material, so REITs/BDCs don't misfire; an unexplained
  gap > 10% is a warning), stub rows anywhere in the history,
  transition periods (< 300 days),
  missing or duplicate periods, bad dates, revenue jumps, share-count changes, equity sign flips,
  unmapped fields (material last year, 0 now) and mixed reporting currencies. Thresholds are
  `DQ_*` in config. Stub periods (income, balance and cash flow) are excluded from indicators
  (`drop_stub_rows`; growth across the gap is NaN) and from the DCF (every dropped period is
  noted; FCF CAGR is measured over elapsed fiscal years). Surfaced on the Stock Analysis page, in the research report (section plus
  a Key Risks line) and as DCF data notes; `store_audit` gives raw counts for the Parquet
  store. Add new deterministic checks here, never as silent corrections.
- **SEC EDGAR** reconciliation is the next data-quality round. It is disabled until the user
  sets `SEC_USER_AGENT` (SEC requires a contact on every request; never default it).
- **Secrets never reach logs.** `get_logger` uses a redacting formatter
  (`apikey=`/`api_key=`/`token=` values → `***`), and client errors re-raise
  `from None` so the raw requests exception (URL incl. key) isn't chained.
- **A "fundamental quality" rating and a "valuation" verdict are different
  questions and must be reported separately, never collapsed into one label.**
  `investment_score.py`'s composite rating (Buy/Hold/Sell) reflects business
  quality/growth/health/momentum ONLY — valuation is scored as a separate
  `valuation` signal and must never be weighted into the composite (it was, at
  35%, until 2026-11-06). The rating does NOT by itself imply the price
  target is above or below the current price (a high-quality compounder can
  legitimately be "Buy" on fundamentals while "Rich" on valuation). Collapsing
  these produced a shipped report reading "Buy, target -36% below price,"
  which is incoherent and was caught and fixed in `research_report.py`'s
  `_valuation_verdict`. Any new report output must keep these conceptually
  separate and reconcile divergence explicitly rather than picking one number.
- FRED (`src/data/fred_client.py`) supplies macro/economic indicators (GDP,
  CPI, unemployment, Treasury yields, Fed funds rate, China rates/CPI/FX).
  `FRED_API_KEY` is optional: without it the client uses FRED's keyless public
  CSV download (same series IDs, full history). Every consumer must degrade
  gracefully when it's unset or unreachable (log a warning, fall back to a
  static default or "unavailable" — never raise out of a report-generation
  path), per the pattern in `dcf.py`'s `live_risk_free_rate`.

- **Starter-tier endpoint facts (verified live 2026-11-05).** Available:
  statements (10y annual, 40 quarters), ratios/key-metrics incl. TTM, daily
  price history, `sector-pe-snapshot`, `industry-pe-snapshot`,
  `historical-sector-pe`, `historical-industry-pe`, `company-screener`,
  `treasury-rates`, `available-industries/sectors/countries`. Premium (402):
  `batch-quote`, `sp500-constituent`, foreign-exchange tickers. Re-verify
  before relying on anything not in this list.
- **Statistical rigor rules (Quant Lab).** Never draw inference from
  correlations of non-stationary levels — ADF-test and use a stationary
  transform. Report every coefficient with its p-value and n. Use HAC
  (time series) or clustered (panels) standard errors, not plain OLS.
  Every forecast is back-tested against a naive benchmark and labelled
  "illustrative" when it doesn't beat it. Aggregate multiples are
  Σ cap ÷ Σ earnings, never an average of ratios. Peer sets from different
  scopes (industry vs sector) are reported side by side, never blended.
  Data-quality screens (e.g. captive-finance interest, duplicate listings,
  survivorship in early years) must be surfaced as caveats, not silently applied.
- **Charts follow one convention** (`src/reporting/theme.py`,
  `src/reporting/charts.py`): one y-axis per plot (no dual axes — use small
  multiples or a common base), fixed categorical colour order (validated
  for colour-blind safety), colour follows the entity, hover on by default.
  The dashboard and the HTML export must use the same figure builders.

## Roles
- **User:** product direction, FMP account/tier decisions, final say on
  portfolio data handling.
- **Claude:** implementation, module design, keeping things simple and
  incremental rather than over-engineering ahead of need.

## Milestones
- [x] Environment setup (`setup.sh`): Node, Claude Code, Aider, Python data stack.
- [x] `src/config.py`: centralized env/config settings.
- [x] `src/utils/logging.py`: shared logger (console + rotating file handler).
- [x] FMP API client (Starter tier, `/stable` endpoints): quotes, profile,
      financial statements, ratios, key metrics, peers, stock news.
- [x] Stock analysis module (valuation/profitability/growth/leverage summary).
- [x] Industry/peer comparison module (PE and financial statement comparison
      across FMP-identified peers; foreign-exchange peers auto-filtered since
      they 402 on Starter tier).
- [x] DCF valuation model (CAPM-derived WACC, two-stage FCF projection +
      Gordon Growth terminal value).
- [x] Resilience: retry/backoff on FMP calls, input validation across client
      and analysis modules.
- [x] Code quality tooling: ruff (lint + format) and mypy, both clean across `src/`.
- [x] Streamlit dashboard: Stock Analysis, Industry Comparison, and DCF
      Valuation pages with charts.
- [x] Portfolio & PnL: SQLite-backed transaction recording (BUY/SELL, validated,
      oversell blocked) with FIFO cost-basis realized PnL and live unrealized
      PnL; Portfolio dashboard page for recording transactions and viewing
      positions/PnL.
- [x] Knowledge log: SQLite-backed quotes/notes (books, Buffett, fundamental
      analysis insights), with author/category/ticker tagging, search, and a
      random "quote of the moment" display; Knowledge Log dashboard page.
- [x] Macro backdrop via FRED (free alternative to FMP's Premium-gated economic
      indicators): GDP growth, CPI YoY, unemployment, Fed funds rate, 10Y
      Treasury yield, yield curve inversion signal. DCF's risk-free rate now
      defaults to the live 10Y Treasury yield (falls back to a static
      assumption if FRED is unavailable).
- [x] Investment scorecard (`investment_score.py`): CFA-style composite score
      (quality, growth, financial health, momentum; valuation reported as a
      separate signal, not in the composite) with an
      auditable banding methodology and a Buy/Hold/Sell rating; usable
      per-stock or to rank/screen a peer group (`rank_opportunities`).
- [x] Comprehensive equity research report (`research_report.py`): executive
      summary, macro backdrop, multi-year financial performance, DCF + peer
      valuation triangulation (blended price target), scorecard breakdown,
      identified risks, and a rating-vs-valuation reconciliation. Research
      Report dashboard page.
- [x] Currency-consistency guard: foreign-ADR quote/reporting-currency
      mismatches (Sony, Panasonic, Samsung, Xiaomi, ...) are detected and
      handled across `stock_analysis.py`, `industry_analysis.py`, and `dcf.py`
      (see Rules — this was a real bug, not a hypothetical).
- [x] Stock Analysis & Industry Comparison rebuilt on time series (2026-11-05): `indicators.py`
      catalog (gearing, net debt/EBITDA, coverage, ROIC, FCF margin, bank efficiency ratio, P/FFO, ...),
      industry-specific indicator profiles, selectable indicators, return vs risk vs valuation reported
      separately, peer filters (scope, cap-size buckets, US-only, include/exclude), industry aggregates,
      $-formatted tables. Annual only: FMP quarterly ratios are Premium (402, verified).
- [x] Data quality, round 1 (2026-11-07): peer funnel and related-industry escalation (fixes
      AAPL's 0 industry peers); statement consistency checks surfaced in the Stock Analysis
      page, the research report and the DCF page; store audit. Next: SEC EDGAR reconciliation.
- [ ] News integration into reports (news endpoint exists in `fmp_client.py`,
      not yet wired into analysis output or the dashboard).
- [x] Quant Lab (2026-11-05): source-agnostic series loader (FRED incl. keyless
      CSV fallback, World Bank, FMP prices / sector & industry P/E history);
      statistical engine (ADF stationarity, correlation with p-values on levels
      AND stationary transforms, lead/lag, Newey-West HAC and company-clustered
      panel OLS, HP-filter cycles + periodogram + OECD phase, back-tested
      forecast vs naive); Macro Lab, Rate Sensitivity and Relative Valuation
      pages; shared Plotly theme and self-contained HTML report export.
- [x] Columnar local data store (2026-11-05): Parquet files (`data/store/`)
      queried with DuckDB; bulk statement sync (`python -m src.data.sync`);
      macro series write-through with offline fallback; Fundamentals Explorer
      page (sector trends, growth leaderboard, company drill-down, read-only
      SQL). Initial load: 10y annual income statements for ~690 US-listed
      companies >= $10B (6.8 MB).

## Architecture
Project root: `/root/projects/stock-research/` (its own git repo — deliberately
*not* the home directory, so `.ssh`, shell history and machine-level credential
scripts like `~/bedrock_setup.sh` are outside the project and the repo).
- `setup.sh` — provisions the dev environment.
- `pyproject.toml` — ruff (lint + format) and mypy configuration.
- `requirements.txt` — runtime dependencies, pinned (`==`); `requirements-dev.txt`
  adds pinned ruff/mypy/type-stubs on top.
- `.gitignore` — excludes `.env*` (except `.env.example`), `data/`, `log/`,
  tool caches, editor swap files.
- `src/config.py` — centralized settings: FMP base URL/key, log dir/level,
  DCF default assumptions, peer-comparison defaults, FMP retry/backoff
  settings. Loads `.env`.
- `src/utils/logging.py` — shared logger (console + rotating file handler).
- `src/data/fmp_client.py` — `FMPClient` wrapping FMP's stable API, with
  retry/backoff and input validation.
- `src/analysis/stock_analysis.py` — single-stock fundamentals summary (`analyze_stock`, used by the
  scorecard / research report).
- `src/analysis/indicators.py` — indicator catalog (~50 indicators tagged return / risk / valuation,
  each with format, direction, ratio-of-sums aggregate definition, currency-guard flag) and
  `load_company` → `CompanyData` (10y annual indicator history, TTM valuation, price return/risk incl.
  HAC beta vs SPY). Industry profiles (`config.INDUSTRY_INDICATOR_PROFILES`) pick default and
  not-meaningful indicators per industry (banks, insurance, REITs, software, semis, retail, energy, utilities).
- `src/analysis/business_model.py` — `classify(profile, income, balance, risk_free_rate)` →
  `BusinessModel(kind, reason, captive_finance, subtype, override)`; drives DCF applicability,
  peer method, peer matching and scorecard health metrics.
- `src/analysis/data_quality.py` — deterministic statement checks (`latest_complete_balance`:
  stub-row fallback with a disclosed note); used by stock_analysis, dcf and relative_valuation.
- `src/analysis/valuation_summary.py` — `summarize_valuation(price, dcf, peer)` →
  `ValuationSummary`: method eligibility, blended / low-precision / indeterminate verdict,
  divergence; shared by `research_report` and `investment_score`.
- `src/analysis/industry_analysis.py` — `run_industry_comparison(symbol, PeerFilters)` → `IndustryComparisonReport`: one peer scope at a time
  (industry → explicit sector fallback when < 3 peers; never blended), cap-size / similar-size / US-only
  filters, ex-subject peer median / mean (non-multiples) / Σ÷Σ aggregate in the peer group's main
  reporting currency, per-year aggregate history, findings per group.
- `src/reporting/formatting.py` — money/price formatting with currency symbol ($ for USD, local otherwise).
- `src/analysis/dcf.py` — discounted cash flow valuation model.
- `src/dashboard/app.py` + `src/dashboard/pages/` — Streamlit multipage app
  (Stock Analysis, Industry Comparison, DCF Valuation). Run with
  `streamlit run src/dashboard/app.py` from the project root (each page file
  bootstraps `sys.path` to the project root itself, so no `PYTHONPATH` setup
  is needed).
- `src/portfolio/db.py` — SQLite connection + schema (`transactions` table).
  DB file lives at `data/portfolio.db` (gitignored — personal financial data).
- `src/portfolio/transactions.py` — `record_buy`/`record_sell`/`list_transactions`,
  with validation (finite quantity/price > 0, ISO date not in the future, oversell
  blocked against the quantity held as of the SELL's date, FIFO timeline-aware).
- `src/portfolio/pnl.py` — FIFO cost-basis engine: `compute_fifo` (per-symbol
  realized trades + remaining open lots), `get_open_positions`,
  `portfolio_summary` (live unrealized PnL via `FMPClient.get_quote` + all-time
  realized PnL totals).
- `src/dashboard/pages/4_Portfolio.py` — record transactions, view positions/PnL,
  view transaction history.
- `src/knowledge/db.py` — SQLite connection + schema (`quotes` table). DB file
  at `data/knowledge.db` (gitignored, same as the portfolio DB).
- `src/knowledge/notes.py` — `add_quote`/`list_quotes` (filter by category,
  author, symbol, or free-text search)/`get_random_quote`/`delete_quote`.
- `src/dashboard/pages/5_Knowledge_Log.py` — add/browse/delete quotes, random
  quote display.
- `src/data/fred_client.py` — `FREDClient` wrapping the FRED economic-data API
  (`get_series`, `get_latest`), same retry/backoff pattern as `fmp_client.py`.
- `src/analysis/macro_analysis.py` — macro backdrop from FRED: GDP growth,
  CPI YoY (computed from the raw index), unemployment, Fed funds rate, 10Y
  Treasury yield, yield-curve-inversion signal.
- `src/analysis/investment_score.py` — fundamental-quality scorecard
  (`compute_investment_score`, plus a separate valuation signal) and peer-group
  screener (`rank_opportunities`; one bad symbol is skipped, never fatal);
  banding methodology and weights live in `src/config.py:SCORE_WEIGHTS` /
  `SCORE_RATING_BANDS`.
- `src/analysis/research_report.py` — full equity research report generator
  (`generate_research_report`/`render_research_report`); blends the credible DCF and
  peer-multiple valuations into one target (or reports "Indeterminate"), keeps the fundamental-quality
  rating and valuation verdict explicitly separate (see Rules).
- `src/dashboard/pages/6_Research_Report.py` — renders the full report.
- `src/utils/cache.py` — on-disk TTL cache for API responses (`data/cache/`,
  gitignored; TTLs in `config.CACHE_TTL_SECONDS`; errors never cached).
- `src/data/worldbank_client.py` — keyless World Bank indicators (annual, global).
- `src/data/series_loader.py` — `load_series`/`load_many`: any catalog key
  (`config.MACRO_SERIES_CATALOG`) or spec (`fred:`, `worldbank:`, `price:`,
  `sectorpe:`, `industrype:`) → uniform `SeriesData`. Add new series to the
  catalog, not to code.
- `src/analysis/timeseries.py` — pure statistical engine (no I/O): align,
  transforms, ADF, descriptive stats, correlation + p-values, rolling corr,
  lead/lag, `ols_hac`, `ols_clustered`, HP filter, `cycle_analysis`,
  back-tested `forecast`.
- `src/analysis/macro_lab.py` — `run_macro_lab`: multi-variable orchestration
  → `MacroLabReport` (data, findings, methodology, caveats, sources).
- `src/analysis/rate_sensitivity.py` — `run_rate_sensitivity`: industry
  effective cost of debt vs a macro rate, fiscal-window alignment, panel
  pass-through, rate-shock scenarios.
- `src/analysis/relative_valuation.py` — `run_relative_valuation`: TTM P/E vs
  industry peers, size-matched sector peers, FMP industry/sector, US large-cap
  market proxy, and point-in-time P/E history; and `peer_valuation` — the single
  peer engine behind the research report's and scorecard's peer-implied price.
- `src/reporting/` — `theme.py` (tokens + Plotly templates, light/dark),
  `charts.py` (figure factories), `builders.py` (tables/figures/KPIs per
  analysis), `html_report.py` (self-contained HTML export).
- `src/dashboard/ui.py` — shared Streamlit UI kit (page chrome, KPI tiles,
  findings/methodology/caveats, themed charts, report download).
- `src/dashboard/app.py` is the `st.navigation` router (Home / Research /
  Quant Lab / Portfolio); `.streamlit/config.toml` holds the app theme.
- `src/dashboard/pages/7_Macro_Lab.py`, `8_Rate_Sensitivity.py`,
  `9_Relative_Valuation.py` — Quant Lab pages.
- `src/data/store.py` — Parquet store + `StoreDB` (DuckDB). Upsert keyed on
  (date, period), atomic writes, ids as strings / numerics as float64.
  `StoreDB(read_only_sql=True)` materialises tables then disables external
  access — the only mode allowed for user-typed SQL (`query_user`).
- `src/data/sync.py` — `build_universe` (screener, share-class dedupe,
  non-equity listings such as baby bonds/preferreds filtered) and
  `sync_statements` (per-company isolation, skip-if-fresh, progress callback).
  CLI: `python -m src.data.sync [--min-cap 2e9] [--kinds income_statement balance_sheet] [--symbols ...]`.
- `src/analysis/fundamentals_panel.py` — company-year ratio panel via DuckDB
  SQL; `sector_trend`, `growth_leaderboard` (small-base CAGR guard),
  `analyze_panel` (findings + methodology + survivorship-bias caveat).
- `src/dashboard/pages/10_Fundamentals_Explorer.py` — panel UI + sync + SQL workbench.
- `log/` — runtime log output (gitignored).
- `data/` — local SQLite database(s) (gitignored — personal financial data).

## Development Workflow
- Lint/format: `ruff check src/` and `ruff format src/` (or `--check` for CI-less
  verification) before considering a change done.
- Type check: `mypy` (uses `pyproject.toml` config, checks the `src` package).
- Run the dashboard: `streamlit run src/dashboard/app.py`.
- Dashboard verification: besides `curl`, drive each changed page with
  `streamlit.testing.v1.AppTest` (submit the form, assert no exceptions) —
  this catches runtime errors a 200 response never shows.
- No automated test suite or CI — deliberate choice for this personal,
  local-only project (see Rules). Verify changes by running the relevant
  module/function directly and, for dashboard changes, a headless
  `streamlit run` + `curl` smoke test of each page.

## Version Control
- Git repo at the project root. `main` is always in a working state (lint/type
  clean, dashboard starts).
- Do non-trivial work on a short-lived branch (`feature/<name>`, `fix/<name>`),
  merge to `main` when the Definition of Done is met.
- Small, focused commits — one logical change each, imperative message
  ("Add FX guard to peer PE"), with a body explaining *why* when not obvious.
- Before every commit: `git status` + `git diff --staged` and confirm nothing
  secret or personal is staged (`.env`, `data/*.db`, logs, API keys in code).
- Never use `git add -A` / `git add .` blindly; never rewrite pushed history.
- Claude commits only when the user asks.

## Review Agents
Independent, read-only reviewers in `.claude/agents/`. They report findings marked
CONFIRMED/PLAUSIBLE with evidence; they never edit code. Fixes are applied separately, then
re-verified by the same reviewer.
- `qa-reviewer`: generalist. Gates, Rules, correctness, security, dashboard. Run it before
  merging any non-trivial change.
- `quant-reviewer`: statistics and econometrics (stationarity, HAC/clustered SEs,
  forecast back-tests, alignment, aggregation). Run it when Quant Lab, timeseries or panel
  code changes.
- `valuation-reviewer`: CFA-style valuation (DCF applicability, WACC, multiples, peer
  sets, rating vs valuation, FIFO PnL) on an adversarial company set. Run it when valuation,
  scoring, report or PnL code changes.
Data cleaning is deliberately **not** an agent: it must be deterministic, auditable code
(e.g. `sync.py` filters, the fiscal-year dedupe in `fundamentals_panel.py`).

## Definition of Done
A change is not "done" — and Claude must not report it as done — until every
applicable item holds:
1. **Lint/format clean:** `ruff check src/` and `ruff format --check src/` pass.
2. **Types clean:** `mypy` passes with no new `# type: ignore`.
3. **Actually executed:** the changed code path was run against real data, not
   just imported. For analysis/valuation changes, at minimum:
   a domestic large-cap (e.g. `AAPL`), a currency-mismatch ADR (e.g. `SONY`),
   and one partial-data case (e.g. a bank with no interest expense).
4. **Dashboard smoke test** (if any page or anything it imports changed):
   headless `streamlit run` + `curl` each affected page, no tracebacks in logs.
5. **Rules respected:** settings in `config.py`, logging via `get_logger`,
   currency check applied to any new price ÷ per-share computation, rating and
   valuation kept separate, input validation at module boundaries.
6. **Docs updated:** Architecture/Milestones in this file reflect new or
   renamed modules; any new env var added (blank) to `.env.example`.
7. **Honest summary:** the final report states what was verified, what was
   not, and any assumption that affects financial output.

## Dependency Management
- All dependencies are pinned to exact versions (`==`) in `requirements.txt`
  (runtime) and `requirements-dev.txt` (tooling). Target Python 3.10.
- Install: `pip install -r requirements-dev.txt`.
- Add a dependency only when needed, pinned, in the correct file, with a
  one-line reason in the commit message.
- Upgrade deliberately, one package (or tightly coupled group) per commit,
  then re-run the full Definition of Done — never a blanket `pip install -U`.
- `matplotlib` and `yfinance` are pinned but not imported anywhere in `src/` —
  remove them, or document their purpose, before relying on them.
