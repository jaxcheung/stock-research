# src/analysis/fundamentals_panel.py
#
# Cross-sectional, multi-year fundamentals from the local Parquet/DuckDB store
# (populated by `python -m src.data.sync`). Answers questions like "how have
# operating margins evolved by sector over 10 years?", "who compounded revenue
# and EPS fastest?", "which sectors carry the heaviest interest burden?".
#
# Methodology:
# - Ratios only across companies (margins, growth, intensities): every ratio
#   is computed inside one company's reporting currency, so mixing USD/CNY/JPY
#   reporters is valid. No cross-company sums of currency amounts.
# - Cross-sectional statistic is the MEDIAN (robust to outliers); cells with
#   fewer than PANEL_MIN_COMPANIES companies are suppressed.
# - Years are calendarised by fiscal-year-end date (FY ending Sep-2025 → 2025),
#   shifted back FISCAL_YEAR_END_GRACE_DAYS so 52/53-week years ending in the first
#   days of January (CDNS, JNJ, LHX, ...) count as the prior year instead of
#   colliding with the previous FY. Any residual duplicate (symbol, year) — e.g.
#   a fiscal-year-end change — keeps the row with revenue, then the later period.
# - Growth uses consecutive fiscal years only; CAGR requires positive endpoints.
# - "Complete" years: those where >= 80% of the universe has reported, so the
#   latest partially-reported year doesn't distort trends.

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import FISCAL_YEAR_END_GRACE_DAYS, LEADERBOARD_MIN_START_REVENUE, PANEL_MIN_COMPANIES
from src.data.store import StoreDB
from src.utils.logging import get_logger

logger = get_logger("fundamentals_panel")

METRICS: dict[str, dict[str, str]] = {
    "grossMargin": {"label": "Gross margin", "fmt": "pct", "kind": "level"},
    "operatingMargin": {"label": "Operating margin", "fmt": "pct", "kind": "level"},
    "netMargin": {"label": "Net margin", "fmt": "pct", "kind": "level"},
    "ebitdaMargin": {"label": "EBITDA margin", "fmt": "pct", "kind": "level"},
    "rdIntensity": {"label": "R&D / revenue", "fmt": "pct", "kind": "level"},
    "sgaIntensity": {"label": "SG&A / revenue", "fmt": "pct", "kind": "level"},
    "interestBurden": {"label": "Interest expense / revenue", "fmt": "pct", "kind": "level"},
    "interestCoverage": {"label": "EBIT / interest expense (×)", "fmt": "x", "kind": "level"},
    "effectiveTaxRate": {"label": "Effective tax rate", "fmt": "pct", "kind": "level"},
    "revenueGrowth": {"label": "Revenue growth (YoY)", "fmt": "pct", "kind": "change"},
    "epsGrowth": {"label": "Diluted EPS growth (YoY)", "fmt": "pct", "kind": "change"},
}

_PANEL_SQL = """
WITH raw AS (
    SELECT
        i.symbol,
        CAST(i.date AS DATE) AS periodEnd,
        year(CAST(i.date AS DATE) - INTERVAL {grace} DAY) AS year,
        i.reportedCurrency AS currency,
        i.revenue, i.grossProfit, i.operatingIncome, i.netIncome, i.ebitda,
        i.researchAndDevelopmentExpenses AS rd,
        i.sellingGeneralAndAdministrativeExpenses AS sga,
        i.interestExpense, i.incomeTaxExpense, i.incomeBeforeTax, i.epsDiluted
    FROM income_annual i
    WHERE i.period = 'FY'
), base AS (
    SELECT * FROM raw
    -- Prefer a full-year row with revenue over a transition stub (LHX FY-2019 stub has revenue 0)
    QUALIFY row_number() OVER (
        PARTITION BY symbol, year ORDER BY (coalesce(revenue, 0) > 0) DESC, periodEnd DESC
    ) = 1
), lagged AS (
    SELECT *,
        lag(revenue) OVER w AS prevRevenue,
        lag(epsDiluted) OVER w AS prevEps,
        lag(year) OVER w AS prevYear
    FROM base
    WINDOW w AS (PARTITION BY symbol ORDER BY periodEnd)
)
SELECT
    l.symbol, l.periodEnd, l.year, l.currency,
    c.companyName AS company, coalesce(c.sector, 'Unclassified') AS sector, c.industry, c.marketCap,
    l.revenue, l.netIncome, l.operatingIncome, l.epsDiluted,
    CASE WHEN l.revenue > 0 THEN l.grossProfit / l.revenue END AS grossMargin,
    CASE WHEN l.revenue > 0 THEN l.operatingIncome / l.revenue END AS operatingMargin,
    CASE WHEN l.revenue > 0 THEN l.netIncome / l.revenue END AS netMargin,
    CASE WHEN l.revenue > 0 THEN l.ebitda / l.revenue END AS ebitdaMargin,
    CASE WHEN l.revenue > 0 AND l.rd > 0 THEN l.rd / l.revenue END AS rdIntensity,
    CASE WHEN l.revenue > 0 AND l.sga > 0 THEN l.sga / l.revenue END AS sgaIntensity,
    CASE WHEN l.revenue > 0 AND l.interestExpense >= 0 THEN l.interestExpense / l.revenue END AS interestBurden,
    CASE WHEN l.interestExpense > 0 THEN l.operatingIncome / l.interestExpense END AS interestCoverage,
    CASE WHEN l.incomeBeforeTax > 0 THEN l.incomeTaxExpense / l.incomeBeforeTax END AS effectiveTaxRate,
    CASE WHEN l.prevYear = l.year - 1 AND l.prevRevenue > 0 THEN l.revenue / l.prevRevenue - 1 END AS revenueGrowth,
    CASE WHEN l.prevYear = l.year - 1 AND l.prevEps > 0 AND l.epsDiluted IS NOT NULL
         THEN l.epsDiluted / l.prevEps - 1 END AS epsGrowth
FROM lagged l
JOIN companies c USING (symbol)  -- universe snapshot defines the scope
"""


@dataclass
class PanelReport:
    panel: pd.DataFrame
    coverage: dict[str, object]
    complete_years: list[int]
    findings: list[str] = field(default_factory=list)
    methodology: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)


def load_panel(db: StoreDB | None = None) -> pd.DataFrame:
    db = db or StoreDB()
    if "income_annual" not in db.tables:
        raise ValueError("No income statements in the local store — run `python -m src.data.sync` first.")
    if "companies" not in db.tables:
        raise ValueError("No company universe in the store — run a sync to snapshot one.")
    df = db.query(_PANEL_SQL.format(grace=int(FISCAL_YEAR_END_GRACE_DAYS)))
    # Winsorise extreme ratios at the 1st/99th percentile per metric (data errors,
    # near-zero denominators) so a single outlier can't dominate a chart scale.
    for m in METRICS:
        if m in df:
            lo, hi = df[m].quantile([0.01, 0.99])
            df[m] = df[m].clip(lo, hi)
    return df


def complete_years(panel: pd.DataFrame, share: float = 0.8) -> list[int]:
    counts = panel.groupby("year")["symbol"].nunique()
    if counts.empty:
        return []
    top = counts.max()
    return sorted(int(y) for y, n in zip(counts.index, counts.to_numpy(), strict=True) if n >= share * top)


def sector_trend(
    panel: pd.DataFrame, metric: str, years: list[int] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(median pivot sector×year, company-count pivot). Cells with < PANEL_MIN_COMPANIES are NaN.
    Includes an 'All companies' row."""
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {list(METRICS)}")
    d = panel.dropna(subset=[metric])
    if years:
        d = d[d["year"].isin(years)]
    med = d.pivot_table(index="sector", columns="year", values=metric, aggfunc="median")
    cnt = d.pivot_table(index="sector", columns="year", values=metric, aggfunc="count")
    allrow = d.groupby("year")[metric].median().rename("All companies")
    allcnt = d.groupby("year")[metric].count().rename("All companies")
    med = pd.concat([med, allrow.to_frame().T])
    cnt = pd.concat([cnt, allcnt.to_frame().T])
    med = med.where(cnt >= PANEL_MIN_COMPANIES).dropna(how="all")
    cnt = cnt.reindex(med.index)
    order = list(med.drop(index="All companies").iloc[:, -1].sort_values(ascending=False).index) + [
        "All companies"
    ]
    return med.reindex(order), cnt.reindex(order)


def growth_leaderboard(
    panel: pd.DataFrame,
    start_year: int,
    end_year: int,
    min_start_revenue: float = LEADERBOARD_MIN_START_REVENUE,
) -> pd.DataFrame:
    """Per company: revenue & EPS CAGR between start and end year (positive
    endpoints only), average and trend of operating margin, margin volatility.
    Companies whose start-year revenue is below `min_start_revenue` are kept
    but flagged `smallBase` and ranked last — their CAGR is not meaningful."""
    if end_year <= start_year:
        raise ValueError("end_year must be after start_year")
    span = end_year - start_year
    d = panel[(panel["year"] >= start_year) & (panel["year"] <= end_year)]
    first = d[d["year"] == start_year].set_index("symbol")
    last = d[d["year"] == end_year].set_index("symbol")
    common = first.index.intersection(last.index)
    rows = []
    for sym in common:
        fst, lst = first.loc[sym], last.loc[sym]
        g = d[d["symbol"] == sym].sort_values("year")
        om = g["operatingMargin"].dropna()
        slope = float(np.polyfit(g.loc[om.index, "year"], om, 1)[0]) if len(om) >= 4 else None
        rows.append(
            {
                "symbol": sym,
                "company": lst["company"],
                "sector": lst["sector"],
                "currency": lst["currency"],
                "revenueStart": fst["revenue"],
                "revenueEnd": lst["revenue"],
                "revenueCAGR": (lst["revenue"] / fst["revenue"]) ** (1 / span) - 1
                if fst["revenue"] > 0 and lst["revenue"] > 0
                else None,
                "epsCAGR": (lst["epsDiluted"] / fst["epsDiluted"]) ** (1 / span) - 1
                if pd.notna(fst["epsDiluted"])
                and pd.notna(lst["epsDiluted"])
                and fst["epsDiluted"] > 0
                and lst["epsDiluted"] > 0
                else None,
                "avgOperatingMargin": float(om.mean()) if not om.empty else None,
                "operatingMarginTrendPpYr": slope * 100 if slope is not None else None,
                "operatingMarginVolatility": float(om.std(ddof=1)) if len(om) > 2 else None,
                "yearsObserved": int(g["year"].nunique()),
                "marketCap": lst["marketCap"],
            }
        )
    lb = pd.DataFrame(rows)
    if lb.empty:
        return lb
    lb["smallBase"] = lb["revenueStart"] < min_start_revenue
    ranked = lb[~lb["smallBase"]]
    lb["revenueCAGRSectorPctile"] = ranked.groupby("sector")["revenueCAGR"].rank(pct=True) * 100
    return lb.sort_values(
        ["smallBase", "revenueCAGR"], ascending=[True, False], na_position="last"
    ).reset_index(drop=True)


def analyze_panel(db: StoreDB | None = None) -> PanelReport:
    panel = load_panel(db)
    yrs = complete_years(panel)
    cov = {
        "companies": int(panel["symbol"].nunique()),
        "companyYears": len(panel),
        "yearRange": f"{int(panel['year'].min())}–{int(panel['year'].max())}" if not panel.empty else "—",
        "completeYears": f"{yrs[0]}–{yrs[-1]}" if yrs else "—",
        "currencies": sorted(panel["currency"].dropna().unique().tolist()),
        "sectors": int(panel["sector"].nunique()),
    }
    rep = PanelReport(panel=panel, coverage=cov, complete_years=yrs)
    if len(yrs) >= 2:
        y0, y1 = yrs[0], yrs[-1]
        med, _ = sector_trend(panel, "operatingMargin", yrs)
        delta = (med[y1] - med[y0]).drop("All companies").dropna().sort_values()
        if not delta.empty:
            rep.findings.append(
                f"Median operating margin across all companies: **{med.loc['All companies', y0]:.1%}** in {y0} → "
                f"**{med.loc['All companies', y1]:.1%}** in {y1}."
            )
            rep.findings.append(
                f"Largest margin expansion {y0}→{y1}: **{delta.index[-1]}** ({delta.iloc[-1] * 100:+.1f} pp); "
                f"largest compression: **{delta.index[0]}** ({delta.iloc[0] * 100:+.1f} pp)."
            )
        g, _ = sector_trend(panel, "revenueGrowth", yrs[1:])
        if not g.empty:
            avg = g.drop("All companies").mean(axis=1).dropna().sort_values()
            if not avg.empty:
                rep.findings.append(
                    f"Fastest-growing sector by median annual revenue growth ({yrs[1]}–{y1}): **{avg.index[-1]}** "
                    f"({avg.iloc[-1]:.1%}/yr); slowest: **{avg.index[0]}** ({avg.iloc[0]:.1%}/yr)."
                )
        ib, _ = sector_trend(panel, "interestBurden", [y1])
        ib_s = ib[y1].drop("All companies").dropna().sort_values()
        if not ib_s.empty:
            rep.findings.append(
                f"Heaviest interest burden in {y1}: **{ib_s.index[-1]}** (median interest expense = "
                f"{ib_s.iloc[-1]:.1%} of revenue) vs {ib_s.index[0]} ({ib_s.iloc[0]:.1%})."
            )
        lb = growth_leaderboard(panel, y0, y1)
        top = lb[~lb["smallBase"]].dropna(subset=["revenueCAGR"]).head(3)
        if not top.empty:
            rep.findings.append(
                f"Top revenue compounders {y0}–{y1}: "
                + ", ".join(f"**{r.symbol}** ({r.revenueCAGR:.0%}/yr)" for r in top.itertuples())
                + "."
            )
    rep.methodology = [
        "**Data** — FMP annual income statements stored locally as Parquet (one file per company, upserted so "
        "restatements replace earlier values) and queried with DuckDB.",
        "**Calendarisation** — each fiscal year is assigned to the calendar year of its fiscal year-end date.",
        "**Ratios** — margins and intensities ÷ revenue (revenue > 0 only); tax rate only where pre-tax income > 0; "
        "coverage only where interest expense > 0. Ratios are winsorised at the 1st/99th percentile.",
        "**Growth** — year-over-year only between consecutive fiscal years; CAGR = (end ÷ start)^(1/years) − 1, "
        "computed only when both endpoints are positive. Companies with start-year revenue below "
        f"{LEADERBOARD_MIN_START_REVENUE / 1e6:,.0f}M (reporting currency) are flagged 'small base' and ranked last.",
        f"**Aggregation** — cross-sectional medians; cells with fewer than {PANEL_MIN_COMPANIES} companies are "
        "suppressed. 'Complete years' = years in which ≥ 80% of the universe has reported.",
        "**Currency** — every ratio is computed within one company's reporting currency; no currency amounts are "
        "summed across companies.",
    ]
    rep.caveats = [
        "**Survivorship bias**: the universe is today's large caps, so the history over-represents winners — "
        "companies that shrank, were acquired or went bankrupt over the decade are absent. Growth and margin "
        "trends are therefore biased upward versus the market of the time.",
        "Financial-sector income statements (banks, insurers) define revenue differently; their gross/operating "
        "margins are not comparable with industrial companies.",
        "Revenue CAGR includes acquisitions — high compounders may be M&A-built rather than organically growing.",
        "FMP's 'interest expense' may be net of capitalised interest and excludes interest booked in cost of revenue "
        "(captive-finance arms).",
    ]
    currencies = [str(c) for c in panel["currency"].dropna().unique()]
    if len(currencies) > 1:
        rep.caveats.append(
            f"Reporting currencies present: {', '.join(sorted(currencies))} — ratio-only comparisons."
        )
    if yrs and int(panel["year"].max()) > yrs[-1]:
        rep.caveats.append(
            f"{int(panel['year'].max())} is only partially reported and excluded from trend findings."
        )
    return rep
