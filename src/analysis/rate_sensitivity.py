# src/analysis/rate_sensitivity.py
#
# Industry debt-cost vs macro-rate analysis, e.g. "how do China's interest
# rates feed through to Chinese automakers' interest expense?"
#
# Per company-year: effective interest rate = interest expense / average
# total debt (opening + closing). Each fiscal year is matched to the macro
# rate AVERAGED over that same 12-month fiscal window (handles non-December
# year ends like Toyota's March FY). Pass-through is estimated on a pooled
# panel of year-over-year changes with company-clustered standard errors,
# then applied to rate-shock scenarios.
#
# Currency: every company ratio is computed inside its own reporting
# currency, so mixing CNY/USD/JPY reporters in RATIOS is valid. Absolute
# amounts are never summed across currencies — the industry aggregate uses
# the median, plus a debt-weighted aggregate only within a single currency.

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.analysis import timeseries as ts
from src.config import (
    RATE_SENSITIVITY_HIGH_SPREAD_FLAG_PP,
    RATE_SENSITIVITY_LOW_RATE_FLAG,
    RATE_SENSITIVITY_MAX_COMPANIES,
    RATE_SENSITIVITY_MIN_COMPANIES,
    RATE_SENSITIVITY_MIN_MARKET_CAP,
    RATE_SHOCK_BPS,
)
from src.data.fmp_client import FMPClient, FMPError, normalize_symbol
from src.data.series_loader import SeriesData, SeriesLoadError, load_series
from src.utils.logging import get_logger

logger = get_logger("rate_sensitivity")


@dataclass
class RateSensitivityReport:
    benchmark: SeriesData
    universe_description: str
    companies: pd.DataFrame  # one row per company (latest year + flags)
    panel: pd.DataFrame  # company-year rows
    industry: pd.DataFrame  # per fiscal year: median / aggregate effective rate, benchmark
    passthrough: dict[str, Any] | None
    passthrough_lagged: dict[str, Any] | None
    scenarios: pd.DataFrame
    findings: list[str] = field(default_factory=list)
    methodology: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)
    scenario_beta: float = 1.0
    scenario_beta_source: str = "full repricing convention"


def _universe(
    client: FMPClient,
    symbols: list[str] | None,
    industry: str | None,
    country: str | None,
    max_companies: int,
) -> tuple[list[dict[str, Any]], str]:
    if symbols:
        out = []
        for s in dict.fromkeys(normalize_symbol(x) for x in symbols):
            try:
                p = client.get_profile(s)
            except FMPError as exc:
                logger.warning("Profile for %s unavailable: %s", s, exc)
                p = {}
            out.append(
                {
                    "symbol": s,
                    "companyName": p.get("companyName", s),
                    "marketCap": p.get("marketCap"),
                    "country": p.get("country"),
                }
            )
        return out[:max_companies], f"{len(out[:max_companies])} user-selected companies"
    if not industry:
        raise ValueError("Provide either symbols or an industry")
    rows = client.screen_companies(
        limit=100,
        industry=industry,
        country=country or None,
        marketCapMoreThan=RATE_SENSITIVITY_MIN_MARKET_CAP,
    )
    # Foreign-exchange listings 402 on Starter — keep US-listed tickers (incl. ADRs).
    rows = [
        r
        for r in rows
        if r.get("symbol") and "." not in r["symbol"] and not r.get("isEtf") and not r.get("isFund")
    ]
    rows = _dedupe_listings(rows)
    rows.sort(key=lambda r: r.get("marketCap") or 0, reverse=True)
    rows = rows[:max_companies]
    desc = (
        f"Top {len(rows)} US-listed issuers (market cap ≥ ${RATE_SENSITIVITY_MIN_MARKET_CAP / 1e9:.0f}B) "
        f"in FMP industry '{industry}'"
    ) + (f", country = {country}" if country else "")
    return rows, desc


_MAJOR_EXCHANGES = {"NASDAQ", "NYSE", "AMEX"}


def _company_key(name: str) -> str:
    n = "".join(ch for ch in name.lower() if ch.isalnum() or ch == " ")
    for suffix in (
        " american depositary shares",
        " co ltd",
        " limited",
        " ltd",
        " inc",
        " corporation",
        " corp",
        " company",
        " group",
        " holdings",
    ):
        n = n.replace(suffix, "")
    return " ".join(n.split())


def _dedupe_listings(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per company. The screener returns the same issuer under several
    tickers (NIO + NIOIF, BYDDY + BYDDF, ...), which would double-count it in
    industry statistics. Prefer the major-exchange listing, then the most liquid.
    """
    best: dict[str, dict[str, Any]] = {}
    for r in rows:
        key = _company_key(r.get("companyName") or r["symbol"])
        rank = (r.get("exchangeShortName") in _MAJOR_EXCHANGES, r.get("volume") or 0)
        cur = best.get(key)
        if cur is None or rank > (cur.get("exchangeShortName") in _MAJOR_EXCHANGES, cur.get("volume") or 0):
            best[key] = r
    return list(best.values())


def _company_years(symbol: str, client: FMPClient, years: int) -> pd.DataFrame:
    inc = client.get_income_statement(symbol, limit=years + 1)
    bal = client.get_balance_sheet(symbol, limit=years + 1)
    if not inc or not bal:
        raise ValueError("no statements returned")
    bal_by_date = {b["date"]: b for b in bal}
    bal_sorted = sorted(bal, key=lambda b: b["date"])
    prior_debt = {
        bal_sorted[i]["date"]: bal_sorted[i - 1].get("totalDebt") for i in range(1, len(bal_sorted))
    }
    rows = []
    for r in inc:
        b = bal_by_date.get(r["date"], {})
        debt_close, debt_open = b.get("totalDebt"), prior_debt.get(r["date"])
        avg_debt = (
            np.nanmean([x for x in (debt_close, debt_open) if x]) if (debt_close or debt_open) else None
        )
        ie = r.get("interestExpense")
        ebit = r.get("operatingIncome")
        revenue = r.get("revenue")
        rows.append(
            {
                "symbol": symbol,
                "date": pd.Timestamp(r["date"]),
                "fiscalYear": r.get("fiscalYear"),
                "currency": r.get("reportedCurrency"),
                "revenue": revenue,
                "ebit": ebit,
                "ebitda": r.get("ebitda"),
                "interestExpense": ie,
                "totalDebt": debt_close,
                "shortTermDebt": b.get("shortTermDebt"),
                "avgDebt": avg_debt,
                "cash": b.get("cashAndShortTermInvestments") or b.get("cashAndCashEquivalents"),
                "effectiveRate": ie / avg_debt * 100 if ie and avg_debt and avg_debt > 0 else None,
                "interestBurden": ie / revenue * 100 if ie and revenue else None,
                "coverage": ebit / ie if ie and ebit is not None and ie > 0 else None,
                "debtToEbitda": debt_close / r["ebitda"]
                if debt_close and r.get("ebitda") and r["ebitda"] > 0
                else None,
            }
        )
    df = pd.DataFrame(rows).sort_values("date")
    return df[df["date"] >= df["date"].max() - pd.DateOffset(years=years)]


def _fiscal_window_mean(series: pd.Series, end: pd.Timestamp) -> float | None:
    window = series[(series.index > end - pd.DateOffset(years=1)) & (series.index <= end)]
    return float(window.mean()) if not window.empty else None


def run_rate_sensitivity(
    benchmark_key: str = "cn_interbank_3m",
    symbols: list[str] | None = None,
    industry: str | None = "Auto - Manufacturers",
    country: str | None = "CN",
    years: int = 9,
    max_companies: int = RATE_SENSITIVITY_MAX_COMPANIES,
    client: FMPClient | None = None,
) -> RateSensitivityReport:
    if years < 2:
        raise ValueError(f"years must be >= 2, got {years}")
    if max_companies < 1:
        raise ValueError(f"max_companies must be >= 1, got {max_companies}")
    client = client or FMPClient()
    try:
        bench = load_series(benchmark_key)
    except SeriesLoadError as exc:
        raise ValueError(f"Benchmark rate unavailable: {exc}") from exc
    if bench.unit != "pct":
        raise ValueError(f"Benchmark {bench.label} is not an interest-rate series (unit {bench.unit})")

    universe, desc = _universe(client, symbols, industry, country, max_companies)
    if not universe:
        raise ValueError("No companies matched the selection")

    frames, skipped = [], {}
    for co in universe:
        try:
            frames.append(_company_years(co["symbol"], client, years))
        except (FMPError, ValueError, KeyError) as exc:
            skipped[co["symbol"]] = str(exc)
            logger.warning("Skipping %s: %s", co["symbol"], exc)
    if not frames:
        raise ValueError(
            "No company had usable statements: " + "; ".join(f"{k}: {v}" for k, v in skipped.items())
        )
    columns = frames[0].columns
    panel = pd.concat([f.dropna(axis=1, how="all") for f in frames], ignore_index=True).reindex(
        columns=columns
    )
    names = {c["symbol"]: c.get("companyName") or c["symbol"] for c in universe}
    panel["company"] = panel["symbol"].map(names)
    panel["benchmark"] = [_fiscal_window_mean(bench.series, d) for d in panel["date"]]
    panel["spread"] = panel["effectiveRate"] - panel["benchmark"]

    # Flag implausibly low effective rates (interest booked outside "interest
    # expense", e.g. captive-finance arms of GM/Ford record it in COGS).
    has_both = panel["effectiveRate"].notna() & panel["benchmark"].notna()
    panel["flagLowRate"] = has_both & (
        panel["effectiveRate"] < RATE_SENSITIVITY_LOW_RATE_FLAG * panel["benchmark"].clip(lower=0.5)
    )
    panel["flagHighRate"] = has_both & (
        panel["effectiveRate"] > panel["benchmark"] + RATE_SENSITIVITY_HIGH_SPREAD_FLAG_PP
    )
    panel["flagged"] = panel["flagLowRate"] | panel["flagHighRate"]
    flagged_companies = set(panel.loc[panel["flagged"], "symbol"])
    # Excluded from inference if flagged in at least 2 of its 3 most recent
    # years. Judged on recent years because the distortion only becomes
    # visible once benchmark rates are meaningfully above zero (GM/Ford look
    # "normal" in 2016-21 at near-zero US rates, implausible since).
    recent_flags = panel.sort_values("date").groupby("symbol").tail(3).groupby("symbol")["flagged"].sum()
    excluded = set(recent_flags[recent_flags >= 2].index)
    clean = panel[~panel["symbol"].isin(excluded) & panel["effectiveRate"].notna()].copy()

    # Industry series per fiscal year.
    clean["year"] = clean["date"].dt.year
    currencies = clean["currency"].dropna().unique()
    ind = clean.groupby("year").agg(
        medianRate=("effectiveRate", "median"),
        q1Rate=("effectiveRate", lambda x: x.quantile(0.25)),
        q3Rate=("effectiveRate", lambda x: x.quantile(0.75)),
        medianCoverage=("coverage", "median"),
        medianBurden=("interestBurden", "median"),
        companies=("symbol", "nunique"),
        benchmark=("benchmark", "mean"),
    )
    if len(currencies) == 1:
        sums = clean.groupby("year")[["interestExpense", "avgDebt"]].sum()
        ind["aggregateRate"] = sums["interestExpense"] / sums["avgDebt"] * 100
    ind["spread"] = ind["medianRate"] - ind["benchmark"]

    # Pass-through: Δ effective rate on Δ benchmark (same year, and 1y lag).
    clean = clean.sort_values(["symbol", "date"])
    clean["dRate"] = clean.groupby("symbol")["effectiveRate"].diff()
    clean["dBench"] = clean.groupby("symbol")["benchmark"].diff()
    clean["dBenchLag1"] = clean.groupby("symbol")["dBench"].shift(1)
    pt = pt_lag = None
    caveats: list[str] = []
    try:
        pt = ts.ols_clustered(clean["dRate"], clean[["dBench"]], clean["symbol"])
    except ValueError as exc:
        caveats.append(f"Pass-through regression not estimated: {exc}")
    try:
        pt_lag = ts.ols_clustered(clean["dRate"], clean[["dBench", "dBenchLag1"]], clean["symbol"])
    except ValueError as exc:
        caveats.append(f"Lagged pass-through not estimated: {exc}")

    # Latest-year company table.
    latest = panel.sort_values("date").groupby("symbol").tail(1).set_index("symbol")
    latest["excluded"] = latest.index.isin(excluded)
    latest["marketCap"] = [
        next((c.get("marketCap") for c in universe if c["symbol"] == s), None) for s in latest.index
    ]

    # Scenarios: (a) near-term — only short-term debt reprices; (b) full
    # repricing × pass-through beta. The estimated beta is used only if it is
    # statistically significant and economically plausible (0-1.5); otherwise
    # the standard stress-test convention of full repricing (beta = 1) applies.
    beta = float(pt["coefficients"].loc["dBench", "coef"]) if pt is not None else float("nan")
    beta_total = beta
    beta_sig = pt is not None and ts.is_significant(float(pt["coefficients"].loc["dBench", "p"]))
    if pt_lag is not None:
        beta_total = float(pt_lag["coefficients"].loc[["dBench", "dBenchLag1"], "coef"].sum())
    scenario_beta = beta_total if beta_sig and 0 <= beta_total <= 1.5 else 1.0
    scenario_beta_source = (
        "estimated cumulative pass-through" if scenario_beta != 1.0 else "full repricing convention"
    )
    scen_rows = []
    for sym, row in latest[~latest["excluded"]].iterrows():
        ie, ebit = row["interestExpense"], row["ebit"]
        if not ie or ie <= 0:
            continue
        for bps in RATE_SHOCK_BPS:
            shock = bps / 10_000
            d_near = (row["shortTermDebt"] or 0) * shock
            d_full = (row["avgDebt"] or 0) * shock * scenario_beta
            scen_rows.append(
                {
                    "symbol": sym,
                    "company": row["company"],
                    "shockBps": bps,
                    "deltaInterestNearTerm": d_near,
                    "deltaInterestFull": d_full,
                    "deltaInterestPctNearTerm": d_near / ie * 100,
                    "deltaInterestPctFull": d_full / ie * 100,
                    "coverageBase": ebit / ie if ebit is not None else None,
                    "coverageFull": ebit / (ie + d_full) if ebit is not None and (ie + d_full) > 0 else None,
                    "deltaEbitMarginPctPts": -d_full / row["revenue"] * 100 if row["revenue"] else None,
                    "currency": row["currency"],
                }
            )
    scenarios = pd.DataFrame(scen_rows)

    rep = RateSensitivityReport(
        benchmark=bench,
        universe_description=desc,
        companies=latest,
        panel=panel,
        industry=ind,
        passthrough=pt,
        passthrough_lagged=pt_lag,
        scenarios=scenarios,
        caveats=caveats,
        skipped=skipped,
    )
    rep.scenario_beta = scenario_beta
    rep.scenario_beta_source = scenario_beta_source
    _findings(rep, excluded, flagged_companies, beta, beta_total, currencies)
    return rep


def _findings(
    rep: RateSensitivityReport,
    excluded: set[str],
    flagged: set[str],
    beta: float,
    beta_total: float,
    currencies: Any,
) -> None:
    ind = rep.industry.dropna(subset=["medianRate"])
    bench_label = rep.benchmark.label
    if not ind.empty and ind["companies"].max() < RATE_SENSITIVITY_MIN_COMPANIES:
        rep.caveats.append(
            f"Industry statistics need at least {RATE_SENSITIVITY_MIN_COMPANIES} companies with usable data "
            f"(at most {int(ind['companies'].max())} here) — a median/IQR of one or two companies is not an "
            "industry reading, so no industry findings are reported. Per-company data is still shown."
        )
        ind = ind.iloc[0:0]
    # Compare like with like: the baseline year is the first in which at least
    # 60% of the universe reports (earlier years hold only the incumbents).
    if not ind.empty:
        min_cos = max(3, int(np.ceil(0.6 * ind["companies"].max())))
        thin = ind[ind["companies"] < min_cos]
        if not thin.empty:
            rep.caveats.append(
                f"FY{thin.index.min()}–FY{thin.index.max()} cover fewer than {min_cos} companies "
                "(later listings or gaps in reported data); they are shown but not used as the comparison baseline."
            )
        ind = ind[ind["companies"] >= min_cos] if (ind["companies"] >= min_cos).any() else ind
    if not ind.empty:
        last_y, first_y = ind.index[-1], ind.index[0]
        last, first = ind.iloc[-1], ind.iloc[0]
        rep.findings.append(
            f"Median effective cost of debt was **{last['medianRate']:.2f}%** in FY{last_y} "
            f"(IQR {last['q1Rate']:.2f}–{last['q3Rate']:.2f}%, {int(last['companies'])} companies) vs "
            f"{first['medianRate']:.2f}% in FY{first_y}; {bench_label} averaged {last['benchmark']:.2f}% over FY{last_y}."
        )
        rep.findings.append(
            f"Industry borrows at a **{last['spread']:+.2f} pp** spread to {bench_label} "
            f"(FY{first_y}: {first['spread']:+.2f} pp) — "
            + (
                "spread has widened: credit risk premium rising."
                if last["spread"] > first["spread"] + 0.25
                else "spread has narrowed: credit conditions easing."
                if last["spread"] < first["spread"] - 0.25
                else "spread broadly stable."
            )
        )
        if pd.notna(last.get("medianCoverage")):
            cov = last["medianCoverage"]
            tone = (
                "comfortable (> 5×)"
                if cov > 5
                else "adequate (2–5×)"
                if cov > 2
                else "thin (< 2×) — refinancing risk"
                if cov > 0
                else "negative — operating losses don't cover interest"
            )
            rep.findings.append(f"Median EBIT interest coverage: **{cov:.1f}×**, {tone}.")
    if rep.passthrough is not None:
        c = rep.passthrough["coefficients"].loc["dBench"]
        sig = ts.is_significant(c["p"])
        rep.findings.append(
            f"Estimated pass-through: a 1 pp move in {bench_label} moves company borrowing costs by "
            f"**{c['coef']:+.2f} pp** in the same fiscal year (p = {c['p']:.3f}, {rep.passthrough['groups']} companies, "
            f"n = {rep.passthrough['n']}) — {'statistically significant' if sig else 'not statistically significant; treat as indicative'}."
            + (
                f" Including a 1-year lag, cumulative pass-through is {beta_total:+.2f} pp."
                if rep.passthrough_lagged is not None
                else ""
            )
        )
    if not rep.scenarios.empty:
        s100 = rep.scenarios[rep.scenarios["shockBps"] == 100]
        if not s100.empty:
            worst = s100.sort_values("deltaInterestPctFull", ascending=False).iloc[0]
            rep.findings.append(
                f"+100 bp scenario (β = {rep.scenario_beta:.2f}, {rep.scenario_beta_source}): median interest expense "
                f"rises {s100['deltaInterestPctFull'].median():.0f}% once all debt reprices "
                f"({s100['deltaInterestPctNearTerm'].median():.0f}% near-term, short-term debt only); most exposed: "
                f"**{worst['company']}** (+{worst['deltaInterestPctFull']:.0f}%)."
            )
    if excluded:
        rep.caveats.append(
            "Excluded from industry statistics (implausible effective rate in 2 of the 3 latest years — below "
            f"{RATE_SENSITIVITY_LOW_RATE_FLAG:.0%} of benchmark, typically captive-finance interest booked in cost of "
            f"revenue, or above benchmark + {RATE_SENSITIVITY_HIGH_SPREAD_FLAG_PP:.0f} pp, typically a tiny debt base): "
            + ", ".join(sorted(excluded))
        )
    partial = flagged - excluded
    if partial:
        rep.caveats.append(
            "Individual years flagged as implausible — too low or too high vs the benchmark (kept): "
            + ", ".join(sorted(partial))
        )
    if len(currencies) > 1:
        rep.caveats.append(
            f"Companies report in multiple currencies ({', '.join(sorted(currencies))}); only within-currency ratios "
            "are compared and the industry figure is a median — no cross-currency sums (see CLAUDE.md currency rule)."
        )
    if rep.benchmark.freq == "A":
        rep.caveats.append(f"{bench_label} is annual; fiscal-window averaging uses the calendar-year value.")
    if rep.skipped:
        rep.caveats.append(
            "Skipped (no usable statements): " + ", ".join(f"{k} ({v})" for k, v in rep.skipped.items())
        )
    rep.caveats.append(
        "Reported 'interest expense' can be net of capitalised interest and excludes lease interest for some "
        "filers; FMP's 'totalDebt' may include lease liabilities. Effective rates are therefore approximate."
    )
    rep.methodology = [
        f"**Universe** — {rep.universe_description}. Foreign-exchange listings are excluded (FMP Starter tier).",
        "**Effective cost of debt** — interest expense ÷ average total debt (opening + closing balance), per fiscal "
        "year, in each company's own reporting currency (so mixed-currency reporters remain comparable as ratios).",
        f"**Benchmark alignment** — {bench_label} ({rep.benchmark.source}) averaged over each company's exact "
        "12-month fiscal window, so March and December year-ends are matched correctly.",
        f"**Data-quality screen** — company-years with an effective rate below {RATE_SENSITIVITY_LOW_RATE_FLAG:.0%} "
        f"of the benchmark or above benchmark + {RATE_SENSITIVITY_HIGH_SPREAD_FLAG_PP:.0f} pp are flagged; companies flagged in 2 of their 3 latest years are excluded from industry "
        "statistics. Duplicate listings of one issuer (ADR + OTC lines) are collapsed to one.",
        "**Industry aggregate** — median and interquartile range across companies per fiscal year; a debt-weighted "
        "aggregate (Σ interest ÷ Σ debt) is shown only when all companies share one reporting currency.",
        "**Pass-through** — pooled panel OLS of Δ effective rate on Δ benchmark (and its 1-year lag), with "
        "company-clustered (CR1) standard errors; first differences remove company fixed effects (credit quality).",
        f"**Scenarios** — shocks of {', '.join(f'{b:+d}' for b in RATE_SHOCK_BPS)} bp. Near-term: only short-term "
        f"debt reprices at the full shock. Full: all average debt reprices × β = {rep.scenario_beta:.2f} "
        f"({rep.scenario_beta_source}; the estimate is used only when significant and within 0–1.5). "
        "Coverage = EBIT ÷ stressed interest expense.",
    ]
