# src/analysis/industry_analysis.py

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from src.analysis.indicators import (
    INDICATORS,
    CompanyData,
    Indicator,
    format_value,
    get_profile,
    load_company,
    load_prices,
    profile_defaults,
)
from src.config import (
    CAP_SIZE_BUCKETS,
    INDICATOR_HISTORY_YEARS,
    INDUSTRY_COMP_MAX_PEERS,
    INDUSTRY_COMP_MAX_PEERS_LIMIT,
    INDUSTRY_COMP_MIN_CAP,
    INDUSTRY_COMP_MIN_FOR_AGGREGATE,
    INDUSTRY_COMP_SCREEN_LIMIT,
    INDUSTRY_COMP_SIMILAR_SIZE_BAND,
    MARKET_BENCHMARK_SYMBOL,
    PEER_SIZE_RELAX_STEPS,
    RELATED_INDUSTRIES,
    RELVAL_MAX_MEANINGFUL_PE,
)
from src.data.fmp_client import FMPClient, FMPError, normalize_symbol
from src.data.sync import is_operating_equity
from src.reporting.formatting import money
from src.utils.logging import get_logger

logger = get_logger("industry_analysis")


def _num(x: Any) -> float:
    """Scalar -> float, NaN for None / non-numeric (keeps pandas-stubs happy)."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


# The legacy FMP-stock-peers comparison (get_peers / compare_companies /
# industry_comparison / peer_aggregate_pe) was removed 2026-11-06: it averaged
# peers across business models (AXP re-rated against V/MA). Price-target peer
# valuation now lives in relative_valuation.peer_valuation.


# =========================================================================== time-series comparison
#
# Peer universe (one scope at a time — industry, sector, FMP peers or a custom
# list are never blended), market-cap / country filters, annual indicator
# panel, ex-subject peer aggregates (median, mean for non-multiples, ratio-of-
# sums aggregate) and price return/risk. Display-time choices (which
# indicators, which years) are applied to the report afterwards, so changing
# them never refetches data.

SCOPES = {
    "industry": "Same FMP industry",
    "sector": "Same FMP sector",
    "fmp": "FMP stock-peers list",
    "custom": "Custom tickers only",
}
SIZE_MODES = {"any": "Any size", "similar": "Similar to subject", "buckets": "Cap-size buckets"}
_MAJOR = ("NASDAQ", "NYSE", "AMEX")
_MCAP_TTM = ("pe", "pb", "ps", "pFcf")  # TTM multiples whose numerator is market cap


@dataclass(frozen=True)
class PeerFilters:
    scope: str = "industry"
    size_mode: str = "similar"
    cap_buckets: tuple[str, ...] = ()
    us_only: bool = False
    max_peers: int = INDUSTRY_COMP_MAX_PEERS
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()

    def validate(self) -> None:
        if self.scope not in SCOPES:
            raise ValueError(f"scope must be one of {list(SCOPES)}, got {self.scope!r}")
        if self.size_mode not in SIZE_MODES:
            raise ValueError(f"size_mode must be one of {list(SIZE_MODES)}, got {self.size_mode!r}")
        if self.size_mode == "buckets" and not self.cap_buckets:
            raise ValueError("Select at least one cap-size bucket (or choose another size filter)")
        unknown = [b for b in self.cap_buckets if b not in CAP_SIZE_BUCKETS]
        if unknown:
            raise ValueError(f"Unknown cap-size bucket(s): {unknown}")
        if not 1 <= self.max_peers <= INDUSTRY_COMP_MAX_PEERS_LIMIT:
            raise ValueError(
                f"max_peers must be between 1 and {INDUSTRY_COMP_MAX_PEERS_LIMIT}, got {self.max_peers}"
            )
        if self.scope == "custom" and not self.include:
            raise ValueError("Custom scope needs at least one ticker in the include list")


def cap_bucket(cap: float | None) -> str:
    if not cap:
        return "—"
    for name, (lo, hi) in CAP_SIZE_BUCKETS.items():
        if cap >= lo and (hi is None or cap < hi):
            return name.split(" ")[0]
    return "—"


def _cap_bounds(
    f: PeerFilters, subject_cap: float | None, lo_mult: float | None = None
) -> tuple[float, float | None]:
    if f.size_mode == "similar" and subject_cap:
        lo, hi = INDUSTRY_COMP_SIMILAR_SIZE_BAND
        lo = lo if lo_mult is None else lo_mult
        return max(subject_cap * lo, INDUSTRY_COMP_MIN_CAP), subject_cap * hi
    if f.size_mode == "buckets":
        ranges = [CAP_SIZE_BUCKETS[b] for b in f.cap_buckets]
        uppers = [r[1] for r in ranges if r[1] is not None]
        upper = None if len(uppers) < len(ranges) else max(uppers)
        return max(min(r[0] for r in ranges), INDUSTRY_COMP_MIN_CAP), upper
    return INDUSTRY_COMP_MIN_CAP, None


def _cap_ok(
    cap: float | None, f: PeerFilters, subject_cap: float | None, lo_mult: float | None = None
) -> bool:
    if not cap:
        return False
    if f.size_mode == "buckets":
        return any(
            cap >= lo and (hi is None or cap < hi) for lo, hi in (CAP_SIZE_BUCKETS[b] for b in f.cap_buckets)
        )
    lo, hi = _cap_bounds(f, subject_cap, lo_mult)
    return cap >= lo and (hi is None or cap <= hi)


def find_peer_candidates(
    subject: CompanyData,
    f: PeerFilters,
    client: FMPClient,
    industries: list[str] | None = None,
    lo_mult: float | None = None,
    funnel: list[dict[str, Any]] | None = None,
) -> list[tuple[str, str, float | None]]:
    """Ordered (symbol, origin, marketCap) candidates for one scope, nearest in
    size first, share classes de-duplicated, US major exchanges only (foreign
    listings 402 on Starter).

    Industry scope screens each of `industries` (default: the subject's own) WITHOUT a
    size band and applies filters client-side, so every stage can be counted into
    `funnel` — the record of why a peer set is thin (AAPL: 8 listed → 0 similar size).
    `lo_mult` overrides the similar-size band's lower bound (mega-cap relaxation)."""
    cands: list[tuple[str, str, float | None]] = []
    cap = subject.market_cap
    if f.scope in ("industry", "sector"):
        lo, hi = _cap_bounds(f, cap, lo_mult)
        # Size band applied SERVER-side: the screener returns at most INDUSTRY_COMP_SCREEN_LIMIT rows,
        # largest first, so an unbanded query silently drops small caps (a $300m bank got peers 2-5x
        # its size). Truncation is recorded in the funnel.
        band: dict[str, Any] = {"marketCapMoreThan": int(lo)}
        if hi is not None:
            band["marketCapLowerThan"] = int(hi)
        if f.scope == "industry":
            pools = [(ind, {"industry": ind, **band}) for ind in (industries or [subject.industry])]
        else:
            pools = [(subject.sector or "", {"sector": subject.sector, **band})]
        rows_all: list[tuple[str, dict[str, Any]]] = []
        for name, filt in pools:
            if f.us_only:
                filt["country"] = "US"
            raw_rows = client.screen_companies(limit=INDUSTRY_COMP_SCREEN_LIMIT, **filt)
            truncated = len(raw_rows) >= INDUSTRY_COMP_SCREEN_LIMIT
            # Drop the subject AND its other share classes (BRK-A for BRK-B, GOOG for GOOGL).
            listed = [
                r
                for r in raw_rows
                if r.get("symbol")
                and r["symbol"] != subject.symbol
                and r.get("companyName") != subject.company
            ]
            eligible = [
                r
                for r in listed
                if r.get("exchangeShortName") in _MAJOR
                and not r.get("isEtf")
                and not r.get("isFund")
                and "." not in r["symbol"]
                and is_operating_equity(r)
            ]
            sized = [r for r in eligible if _cap_ok(r.get("marketCap"), f, cap, lo_mult)]
            if funnel is not None:
                funnel.append(
                    {
                        "pool": name,
                        "size band": f"{money(lo)} – {money(hi) if hi is not None else 'no cap'}",
                        "listed in band (ex-subject)": len(listed),
                        "truncated at screener limit": truncated,
                        "US major exchange, operating equity": len(eligible),
                        "kept": len(sized),
                        # Proxy for "reports in a comparable currency": same country as the subject.
                        # TM (JPY) and RACE (EUR) used to count toward TSLA's minimum of three.
                        "same country": sum(1 for r in sized if r.get("country") == subject.country),
                        "largest kept": ", ".join(
                            r["symbol"] for r in sorted(sized, key=lambda r: -(r.get("marketCap") or 0))[:3]
                        ),
                    }
                )
            rows_all += [(name, r) for r in sized]
        by_name: dict[str, tuple[str, dict[str, Any]]] = {}
        for name, r in rows_all:  # share classes (GOOG/GOOGL): keep the most liquid
            key = r.get("companyName") or r["symbol"]
            if key not in by_name or (r.get("volume") or 0) > (by_name[key][1].get("volume") or 0):
                by_name[key] = (name, r)
        ref = cap or 1.0
        ordered = sorted(by_name.values(), key=lambda nr: abs(np.log((nr[1].get("marketCap") or 1) / ref)))
        own = subject.industry if f.scope == "industry" else None
        cands = [
            (
                r["symbol"],
                SCOPES[f.scope] if (f.scope == "sector" or name == own) else f"Related industry: {name}",
                r.get("marketCap"),
            )
            for name, r in ordered
        ]
    elif f.scope == "fmp":
        for p in client.get_stock_peers(subject.symbol):
            sym = p.get("symbol")
            if sym and "." not in sym and _cap_ok(p.get("mktCap"), f, cap):
                cands.append((sym, SCOPES["fmp"], p.get("mktCap")))
    excluded = {subject.symbol, *f.exclude}
    seen: set[str] = set()
    out: list[tuple[str, str, float | None]] = []
    for sym in f.include:  # explicit includes always kept, first
        if sym not in excluded and sym not in seen:
            seen.add(sym)
            out.append((sym, "Included manually", None))
    for sym, origin, mc in cands:
        if sym not in excluded and sym not in seen:
            seen.add(sym)
            out.append((sym, origin, mc))
    n_forced = len([o for o in out if o[1] == "Included manually"])
    return out[: max(f.max_peers, n_forced)]


@dataclass
class IndustryComparisonReport:
    symbol: str
    subject: CompanyData
    companies: dict[str, CompanyData]  # subject first, then peers in candidate order
    origins: dict[str, str]
    profile_key: str
    profile: dict[str, Any]
    filters: PeerFilters
    base_currency: str | None  # reporting currency used for ratio-of-sums aggregates
    bench_symbol: str
    bench_prices: pd.Series
    skipped: list[tuple[str, str]]
    methodology: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    notice: str | None = None  # scope fallback, shown prominently
    scope_label: str = ""  # the peer scope actually used ("Industry + related: ..." etc.)
    peer_funnel: list[dict[str, Any]] = field(default_factory=list)  # why the peer set is what it is

    @property
    def peers(self) -> list[str]:
        return [s for s in self.companies if s != self.symbol]

    @property
    def defaults(self) -> dict[str, list[str]]:
        return profile_defaults(self.profile)

    @property
    def not_meaningful(self) -> list[str]:
        return [k for k in self.profile.get("not_meaningful", []) if k in INDICATORS]


def run_industry_comparison(
    symbol: str,
    filters: PeerFilters | None = None,
    client: FMPClient | None = None,
    years: int = INDICATOR_HISTORY_YEARS,
) -> IndustryComparisonReport:
    """Load subject + peers (annual indicators, TTM valuation, prices).
    One failing peer is skipped, never fatal; a failing subject raises."""
    f = filters or PeerFilters()
    f.validate()
    symbol = normalize_symbol(symbol)
    f = PeerFilters(
        scope=f.scope,
        size_mode=f.size_mode,
        cap_buckets=f.cap_buckets,
        us_only=f.us_only,
        max_peers=f.max_peers,
        include=tuple(dict.fromkeys(normalize_symbol(s) for s in f.include)),
        exclude=tuple(dict.fromkeys(normalize_symbol(s) for s in f.exclude)),
    )
    client = client or FMPClient()
    bench = load_prices(MARKET_BENCHMARK_SYMBOL, client)
    subject = load_company(symbol, client, years=years, bench_prices=bench)
    profile_key, profile = get_profile(subject.sector, subject.industry)

    funnel: list[dict[str, Any]] = []
    candidates: list[tuple[str, str, float | None]] = []
    notice = None
    scope_label = SCOPES[f.scope]
    used_floor: float | None = None
    if f.scope == "industry":
        # Escalation, each step labelled and never blended: own industry → relaxed size band
        # (mega-caps) → industry + related industries (config) → whole sector. A step succeeds
        # only with enough SAME-COUNTRY peers (a proxy for a comparable reporting currency).
        related = RELATED_INDUSTRIES.get(subject.industry or "", {}).get("related", [])
        steps = PEER_SIZE_RELAX_STEPS if f.size_mode == "similar" else (None,)
        attempts = [([subject.industry], "Same FMP industry")]
        if related:
            attempts.append(([subject.industry, *related], "Industry + related: " + ", ".join(related)))
        found = False
        best: tuple[int, list[tuple[str, str, float | None]], str, float | None] = (-1, [], "", None)
        for industries, label in attempts:
            for lo_mult in steps:
                step_funnel: list[dict[str, Any]] = []
                try:
                    candidates = find_peer_candidates(subject, f, client, industries, lo_mult, step_funnel)
                except FMPError as exc:
                    logger.warning("Peer screen failed for %s: %s", symbol, exc)
                    candidates = [(s_, "Included manually", None) for s_ in f.include]
                floor = _cap_bounds(f, subject.market_cap, lo_mult)[0]
                step = label + (
                    f" (size ≥ {lo_mult:g}× subject = {money(floor)})" if lo_mult is not None else ""
                )
                for row in step_funnel:
                    row["attempt"] = step
                funnel += step_funnel
                comparable = sum(int(r.get("same country", 0)) for r in step_funnel) + len(
                    [x for x in candidates if x[1] == "Included manually"]
                )
                if comparable > best[0]:
                    best = (comparable, candidates, step, floor if lo_mult is not None else None)
                if comparable >= INDUSTRY_COMP_MIN_FOR_AGGREGATE:
                    found, scope_label, used_floor = True, step, (floor if lo_mult is not None else None)
                    break
            if found:
                break
        own_kept = [x for x in candidates if x[1] == SCOPES["industry"]]
        if found and scope_label.startswith("Industry + related") and not own_kept:
            scope_label = scope_label.replace(
                "Industry + related", "Related industries only (none in own industry)"
            )
        if found and len(funnel) > 1:
            first = funnel[0]
            cause = (
                f"no peers within the similar-size band ({first['size band']})"
                if first["listed in band (ex-subject)"] == 0
                else f"too few same-country peers within {first['size band']}"
            )
            notice = f"FMP industry '{subject.industry}': {cause}. Peer set used: {scope_label}" + (
                " — the 'similar size' intent is relaxed; see the peer funnel." if used_floor else "."
            )
        elif not found:
            notice = (
                f"Too few same-country peers in industry '{subject.industry}'"
                + (" and its related industries" if related else "")
                + f" even after relaxing the size band (see the peer funnel), so the peer set uses the whole "
                f"sector '{subject.sector}' instead (similar-size band, same country filter)."
            )
            f = replace(f, scope="sector")
            scope_label = SCOPES["sector"]
            try:
                sector_funnel: list[dict[str, Any]] = []
                candidates = find_peer_candidates(subject, f, client, funnel=sector_funnel)
                for row in sector_funnel:
                    row["attempt"] = "Whole sector"
                funnel += sector_funnel
                sector_comparable = sum(int(r.get("same country", 0)) for r in sector_funnel)
                if sector_comparable < best[0]:
                    # The sector (un-relaxed size band) found fewer comparable peers than the best
                    # industry attempt (GOOGL: 1 vs 2) — keep the industry attempt, and say so.
                    candidates, scope_label, used_floor = best[1], best[2], best[3]
                    f = replace(f, scope="industry")
                    notice = (
                        f"Too few same-country peers in industry '{subject.industry}' even after relaxing the "
                        f"size band, and the whole sector found fewer still ({sector_comparable}); using the "
                        f"best industry attempt — {scope_label} — with only {best[0]} comparable peer(s). "
                        "Treat peer statistics as thin."
                    )
            except FMPError as exc:
                logger.warning("Sector screen failed for %s: %s", symbol, exc)
        if any(r.get("truncated at screener limit") for r in funnel):
            notice = (notice + " " if notice else "") + (
                f"At least one screen hit the {INDUSTRY_COMP_SCREEN_LIMIT}-row screener limit, so some "
                "candidates in the size band may be missing."
            )
        if notice:
            logger.info("%s: %s", symbol, notice)
    else:
        try:
            candidates = find_peer_candidates(subject, f, client, funnel=funnel)
        except FMPError as exc:
            logger.warning("Peer screen failed for %s: %s", symbol, exc)
            candidates = [(s_, "Included manually", None) for s_ in f.include]

    companies: dict[str, CompanyData] = {symbol: subject}
    origins = {symbol: "Subject"}
    skipped: list[tuple[str, str]] = []
    for sym, origin, _ in candidates:
        try:
            c = load_company(sym, client, years=years, bench_prices=bench)
        except (FMPError, ValueError) as exc:
            logger.warning("Skipping peer %s: %s", sym, exc)
            skipped.append((sym, str(exc)))
            continue
        if f.us_only and origin != "Included manually" and c.country != "US":
            skipped.append((sym, f"domiciled in {c.country} (US-only filter)"))
            continue
        companies[sym] = c
        origins[sym] = origin

    # Ratio-of-sums aggregates need one currency: the peer group's most common
    # reporting currency (subject's on a tie), so a JPY-reporting ADR subject
    # doesn't knock every USD peer out of the sums.
    ccys = pd.Series(
        [c.reporting_currency for s, c in companies.items() if s != symbol and c.reporting_currency]
    )
    base_ccy = subject.reporting_currency
    if not ccys.empty:
        counts = ccys.value_counts()
        top = counts[counts == counts.max()].index.tolist()
        base_ccy = base_ccy if base_ccy in top else str(top[0])
    rep = IndustryComparisonReport(
        symbol=symbol,
        subject=subject,
        companies=companies,
        origins=origins,
        profile_key=profile_key,
        profile=profile,
        filters=f,
        base_currency=base_ccy,
        bench_symbol=MARKET_BENCHMARK_SYMBOL,
        bench_prices=bench,
        skipped=skipped,
    )
    rep.notice = notice
    rep.scope_label = scope_label
    rep.peer_funnel = funnel
    rep.methodology = _methodology(rep)
    rep.caveats = _caveats(rep)
    if notice:
        rep.caveats.insert(0, notice)
    return rep


# --------------------------------------------------------------------------- tables & aggregates


def peers_table(rep: IndustryComparisonReport) -> pd.DataFrame:
    rows = []
    for sym, c in rep.companies.items():
        rows.append(
            {
                "symbol": sym,
                "company": c.company,
                "industry": c.industry,
                "country": c.country,
                "marketCap": c.market_cap,
                "capBucket": cap_bucket(c.market_cap),
                "reportingCurrency": c.reporting_currency,
                "currencyMismatch": c.currency_mismatch,
                "latestFY": c.latest_year,
                "origin": rep.origins.get(sym, ""),
            }
        )
    return pd.DataFrame(rows).set_index("symbol")


def snapshot(rep: IndustryComparisonReport, keys: list[str], basis: str = "latest") -> pd.DataFrame:
    """symbol × indicator: latest fiscal year for annual indicators, price
    metrics for market ones, and (basis="ttm") current TTM multiples where
    available for valuation indicators."""
    out: dict[str, dict[str, Any]] = {}
    for sym, c in rep.companies.items():
        row: dict[str, Any] = {}
        last = c.values.iloc[-1] if not c.values.empty else pd.Series(dtype=float)
        for k in keys:
            ind = INDICATORS[k]
            if ind.market:
                row[k] = c.market.get(k)
            elif basis == "ttm" and k in c.ttm:
                row[k] = c.ttm[k]
            else:
                row[k] = last.get(k)
        out[sym] = row
    return pd.DataFrame.from_dict(out, orient="index").astype(float)


def _meaningful_values(s: pd.Series, ind: Indicator) -> pd.Series:
    s = s.dropna()
    if ind.positive_only:
        s = s[s > 0]
        if ind.key == "pe":
            s = s[s <= RELVAL_MAX_MEANINGFUL_PE]
    return s


def _ratio_of_sums(
    rep: IndustryComparisonReport, ind: Indicator, symbols: list[str], year: int | None, basis: str
) -> tuple[float | None, int]:
    """Σ numerator ÷ Σ denominator over `symbols` reporting in the base currency."""
    num_tot = den_tot = 0.0
    n = 0
    for sym in symbols:
        c = rep.companies[sym]
        if basis == "ttm" and ind.key in c.ttm:
            mult, cap = c.ttm.get(ind.key), c.market_cap
            if mult in (None, 0) or not cap or c.currency_mismatch:
                continue
            if ind.key in _MCAP_TTM:
                num, den = cap, cap / mult  # implied earnings / book / sales / FCF (sign kept)
            elif ind.key == "dividendYield":
                num, den = cap * mult, cap
            else:
                continue
        else:
            if ind.agg is None or c.reporting_currency != rep.base_currency:
                continue
            raw = c.raw
            yr = year if year is not None else c.latest_year
            if yr not in raw.index:
                continue
            r = raw.loc[yr]
            num = _num(r.get(ind.agg[0]))
            den = _num(r.get(ind.agg[1]))
            if not (np.isfinite(num) and np.isfinite(den)):
                continue
            if ind.price_derived and c.currency_mismatch:
                continue
        num_tot += float(num)
        den_tot += float(den)
        n += 1
    if n < INDUSTRY_COMP_MIN_FOR_AGGREGATE or den_tot == 0 or (ind.positive_only and den_tot < 0):
        return None, n
    return num_tot / den_tot, n


def peer_aggregates(rep: IndustryComparisonReport, keys: list[str], basis: str = "latest") -> pd.DataFrame:
    """Per indicator, over peers EXCLUDING the subject: median, mean (not for
    multiples — averaging ratios is biased; use the aggregate), ratio-of-sums
    aggregate, n, and the subject's value / percentile vs peers."""
    snap = snapshot(rep, keys, basis)
    peers = rep.peers
    rows = []
    for k in keys:
        ind = INDICATORS[k]
        pv = _meaningful_values(snap.loc[peers, k], ind) if peers else pd.Series(dtype=float)
        enough = len(pv) >= INDUSTRY_COMP_MIN_FOR_AGGREGATE
        agg, n_agg = (None, 0)
        if ind.agg is not None or (basis == "ttm" and k in ("pe", "pb", "ps", "pFcf", "dividendYield")):
            agg, n_agg = _ratio_of_sums(rep, ind, peers, None, basis)
        subj = _num(snap.at[rep.symbol, k])
        pct = None
        directional = ind.higher_is_better is not None
        if directional and np.isfinite(subj) and enough and not (ind.positive_only and subj <= 0):
            pct = float((pv > subj).mean()) if ind.higher_is_better is False else float((pv < subj).mean())
        rows.append(
            {
                "key": k,
                "label": ind.label,
                "group": ind.group,
                "fmt": ind.fmt,
                "subject": subj if np.isfinite(subj) else None,
                "median": float(pv.median()) if enough else None,
                "mean": float(pv.mean()) if enough and ind.fmt != "x" else None,
                "aggregate": agg,
                "nAggregate": n_agg,
                "n": int(len(pv)),
                "percentile": pct,
                "higherIsBetter": ind.higher_is_better,
            }
        )
    return pd.DataFrame(rows).set_index("key")


def indicator_history(rep: IndustryComparisonReport, key: str) -> pd.DataFrame:
    """fiscalYear × symbol for one annual indicator."""
    cols = {s: c.values[key] for s, c in rep.companies.items() if key in c.values}
    return pd.DataFrame(cols).sort_index()


def aggregate_history(rep: IndustryComparisonReport, key: str) -> pd.DataFrame:
    """Per fiscal year: peer median (ex-subject), ratio-of-sums aggregate and n."""
    ind = INDICATORS[key]
    hist = indicator_history(rep, key)
    peers = [p for p in rep.peers if p in hist]
    rows = []
    for yr in hist.index:
        pv = (
            _meaningful_values(pd.Series(hist.loc[yr, peers], dtype=float), ind)
            if peers
            else pd.Series(dtype=float)
        )
        agg, n_agg = _ratio_of_sums(rep, ind, peers, int(yr), "latest") if ind.agg else (None, 0)
        enough = len(pv) >= INDUSTRY_COMP_MIN_FOR_AGGREGATE
        rows.append(
            {
                "fiscalYear": int(yr),
                "median": float(pv.median()) if enough else None,
                "aggregate": agg,
                "n": int(len(pv)),
                "nAggregate": n_agg,
            }
        )
    return pd.DataFrame(rows).set_index("fiscalYear")


def price_table(rep: IndustryComparisonReport) -> pd.DataFrame:
    rows = {
        s: {
            k: c.market.get(k)
            for k in ("ret1y", "cagr3y", "cagr5y", "volatility", "maxDrawdown", "beta", "betaP", "betaN")
        }
        for s, c in rep.companies.items()
    }
    return pd.DataFrame.from_dict(rows, orient="index")


def price_panel(rep: IndustryComparisonReport) -> pd.DataFrame:
    cols = {s: c.prices for s, c in rep.companies.items() if not c.prices.empty}
    if not rep.bench_prices.empty:
        cols[f"{rep.bench_symbol} (market)"] = rep.bench_prices
    return pd.DataFrame(cols).sort_index()


def comparison_findings(rep: IndustryComparisonReport, keys: list[str], basis: str = "ttm") -> list[str]:
    """Where the subject stands vs its peers, per group (return / risk /
    valuation reported separately — never folded into one verdict)."""
    if not rep.peers:
        return ["No peers matched the filters — widen the scope or the size filter."]
    agg = peer_aggregates(rep, keys, basis)
    out: list[str] = []
    for group, label in (("return", "Return"), ("risk", "Risk"), ("valuation", "Valuation")):
        g = agg[(agg["group"] == group) & agg["percentile"].notna() & agg["higherIsBetter"].notna()]
        if g.empty:
            continue
        best = g.sort_values("percentile", ascending=False).iloc[0]
        worst = g.sort_values("percentile").iloc[0]
        word = {"return": "strongest", "risk": "safest", "valuation": "cheapest"}[group]
        detail = (
            f"({format_value(best['subject'], best['fmt'])} vs peer median {format_value(best['median'], best['fmt'])}; "
            f"better than {best['percentile']:.0%} of {int(best['n'])} peers)"
        )
        if best["percentile"] >= 0.5:
            out.append(
                f"**{label}** — {rep.symbol}'s {word} relative position is **{best['label']}** {detail}."
            )
        else:
            # Even the best-ranked metric is below the peer median — don't call it a strength.
            out.append(
                f"**{label}** — {rep.symbol} ranks below the peer median on every selected {group} metric; "
                f"the least unfavourable is **{best['label']}** {detail}."
            )
        if worst.name != best.name and worst["percentile"] < 0.5:
            out.append(
                f"**{label}** — weakest relative position: **{worst['label']}** "
                f"({format_value(worst['subject'], worst['fmt'])} vs median {format_value(worst['median'], worst['fmt'])}; "
                f"better than only {worst['percentile']:.0%} of peers)."
            )
    s, a = (
        (_num(agg.at["pe", "subject"]), _num(agg.at["pe", "aggregate"])) if "pe" in agg.index else (0.0, 0.0)
    )
    if np.isfinite(s) and np.isfinite(a) and s > 0 and a > 0:
        out.append(
            f"P/E ({'TTM' if basis == 'ttm' else 'latest FY'}) of **{s:.1f}×** vs peer aggregate "
            f"(Σ market cap ÷ Σ earnings) of **{a:.1f}×** — a {abs(s / a - 1):.0%} "
            f"{'premium' if s >= a else 'discount'}. "
            "This is a valuation reading, not a quality rating."
        )
    if rep.subject.currency_mismatch:
        out.append(
            f"⚠️ {rep.symbol} quotes in {rep.subject.quote_currency} but reports in "
            f"{rep.subject.reporting_currency}: its valuation multiples are unavailable (no FX conversion)."
        )
    return out


def _methodology(rep: IndustryComparisonReport) -> list[str]:
    f = rep.filters
    size = {
        "any": f"any size ≥ ${INDUSTRY_COMP_MIN_CAP / 1e6:,.0f}M",
        "similar": f"{INDUSTRY_COMP_SIMILAR_SIZE_BAND[0]}×–{INDUSTRY_COMP_SIMILAR_SIZE_BAND[1]}× the subject's market cap",
        "buckets": ", ".join(f.cap_buckets),
    }[f.size_mode]
    used = rep.scope_label or SCOPES[f.scope]
    if "size ≥" in used:
        size = "lower size bound relaxed as shown (upper bound 5× the subject's market cap)"
    return [
        f"Peer universe: **{used}** "
        f"({rep.subject.industry if f.scope == 'industry' else rep.subject.sector}), "
        f"{size}{', US-domiciled only' if f.us_only else ''}; nearest in size first, the subject's own share classes "
        f"and duplicate share classes removed, up to {f.max_peers} peers. Scopes are never blended; the peer "
        "funnel shows each attempt.",
        f"Indicator profile: **{rep.profile['label']}** — default indicators and 'not meaningful' flags are "
        "industry-specific (config.INDUSTRY_INDICATOR_PROFILES).",
        "Annual indicators are computed from FMP annual statements per fiscal year; companies' fiscal years end "
        "in different months, so a 'year' compares fiscal years labelled by FMP, not identical calendar windows.",
        "Peer statistics EXCLUDE the subject. Median uses meaningful values only (multiples ≤ 0, and P/E above "
        f"{RELVAL_MAX_MEANINGFUL_PE:.0f}×, are excluded). A simple mean is shown for non-multiples only.",
        "Aggregate = Σ numerator ÷ Σ denominator (e.g. Σ market cap ÷ Σ net income for P/E) — the index method; "
        "loss-makers enter the sums. Only companies reporting in the peer group's main currency "
        f"({rep.base_currency}) are summed. Needs ≥ {INDUSTRY_COMP_MIN_FOR_AGGREGATE} companies.",
        "Current valuation uses FMP TTM ratios with today's market cap; time series use fiscal-year-end values.",
        f"Price return/risk from daily closes: 1y return, 3y/5y CAGR, annualised volatility, max drawdown, and beta "
        f"vs {rep.bench_symbol} from weekly log returns (OLS, Newey-West HAC SEs; p-value and n reported).",
        "Percentile = share of peers the subject beats, direction-adjusted (lower is better for leverage, "
        "multiples, volatility).",
    ]


def _caveats(rep: IndustryComparisonReport) -> list[str]:
    out = [
        "Survivorship: the peer set is today's screener universe, so historical peer medians exclude companies "
        "that were delisted, acquired or shrank below the size filter.",
        "Price returns exclude dividends (FMP light price history) — total return is higher for dividend payers.",
    ]
    mism = [s for s, c in rep.companies.items() if c.currency_mismatch]
    if mism:
        out.append(
            f"Currency mismatch (quote ≠ reporting currency): {', '.join(mism)} — valuation multiples nulled; "
            "their margins, returns and leverage are still compared."
        )
    other = [s for s in rep.peers if rep.companies[s].reporting_currency != rep.base_currency]
    if other:
        out.append(
            f"Reporting currency differs from {rep.base_currency} for {', '.join(other)} — excluded from "
            "ratio-of-sums aggregates (sums across currencies are meaningless) but kept in medians."
        )
    if len(rep.peers) < INDUSTRY_COMP_MIN_FOR_AGGREGATE:
        out.append(
            f"Only {len(rep.peers)} peer(s) — fewer than {INDUSTRY_COMP_MIN_FOR_AGGREGATE}, so peer statistics are "
            "not reported. Widen the scope or size filter."
        )
    if rep.skipped:
        out.append("Skipped: " + "; ".join(f"{s} ({r[:60]})" for s, r in rep.skipped[:8]))
    for s, c in rep.companies.items():
        for n in c.notes:
            out.append(f"{s}: {n}.")
    return out
