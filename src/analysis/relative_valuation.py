# src/analysis/relative_valuation.py
#
# Relative valuation: a stock's P/E vs (1) its peer group, (2) its FMP
# industry, (3) its FMP sector, and (4) the US large-cap market — today and
# through time — with growth/profitability context to judge whether a
# premium is earned.
#
# Methodology highlights:
# - P/E is TTM (trailing twelve months). Negative or > RELVAL_MAX_MEANINGFUL_PE
#   multiples are "not meaningful" (NM) and excluded from medians, but loss-
#   makers ARE included in aggregate P/E (Σ market cap ÷ Σ earnings), which is
#   how index P/Es are built — averaging P/E ratios is biased upward.
# - Currency guard (CLAUDE.md): a company whose quote currency differs from its
#   reporting currency has P/E nulled and is excluded; for the subject stock
#   the whole analysis is refused, as every number would be corrupted.
# - Historical P/E uses diluted EPS summed over the last four quarters *as of
#   each quarter's filing date* — no look-ahead bias.
# - The output is a RELATIVE VALUATION verdict (premium/in line/discount). It
#   is deliberately not a Buy/Sell rating (see CLAUDE.md: quality rating and
#   valuation verdict are separate questions).

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from src.analysis.business_model import classify
from src.config import (
    DCF_TERMINAL_GROWTH_RATE,
    PEER_EPS_TTM_DIVERGENCE,
    PEER_METHOD_BY_KIND,
    PEER_ROE_GAP_CAVEAT_PP,
    PEER_TARGET_MIN_PEERS,
    RELVAL_HISTORY_YEARS,
    RELVAL_MARKET_PROXY_MIN_CAP,
    RELVAL_MARKET_PROXY_SIZE,
    RELVAL_MAX_MEANINGFUL_PE,
    RELVAL_MAX_PEERS,
    RELVAL_MIN_MEANINGFUL_PEERS,
    RELVAL_SECTOR_PEERS,
)
from src.data.fmp_client import FMPClient, FMPError, normalize_symbol
from src.data.sync import is_operating_equity
from src.utils.cache import cache_get, cache_set, is_miss, ttl_for
from src.utils.logging import get_logger

logger = get_logger("relative_valuation")

_SNAPSHOT_EXCHANGES = ("NASDAQ", "NYSE")


@dataclass
class Benchmark:
    name: str  # "Peer median", "Industry (FMP)", ...
    pe: float | None
    premium: float | None  # stock PE / benchmark - 1
    basis: str  # how it was computed
    n: int | None = None


@dataclass
class RelativeValuationReport:
    symbol: str
    company: str
    sector: str
    industry: str
    exchange: str
    price: float | None
    stock: dict[str, Any]
    peers: pd.DataFrame
    benchmarks: list[Benchmark]
    market_constituents: pd.DataFrame
    history: pd.DataFrame  # date, stockPE, industryPE, sectorPE, relativeToIndustry
    history_stats: dict[str, Any]
    verdict: str
    verdict_detail: str
    findings: list[str] = field(default_factory=list)
    methodology: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    primary_benchmark: str = "Industry peers — median"


# --------------------------------------------------------------------------- per-company


def _meaningful(pe: float | None) -> bool:
    return pe is not None and not np.isnan(pe) and 0 < pe <= RELVAL_MAX_MEANINGFUL_PE


def _company_row(symbol: str, client: FMPClient) -> dict[str, Any] | None:
    try:
        profile = client.get_profile(symbol)
        ttm = client.get_ratios_ttm(symbol)
        km = client.get_key_metrics_ttm(symbol)
        inc = client.get_income_statement(symbol, limit=2)
    except (FMPError, ValueError) as exc:
        logger.warning("Skipping %s: %s", symbol, exc)
        return None
    if not profile:
        return None
    reporting = inc[0].get("reportedCurrency") if inc else None
    quote_ccy = profile.get("currency")
    mismatch = bool(quote_ccy and reporting and quote_ccy != reporting)
    pe = None if mismatch else ttm.get("priceToEarningsRatioTTM")
    mcap = profile.get("marketCap")
    model = classify(profile, inc[0] if inc else None)
    pb = None if mismatch else ttm.get("priceToBookRatioTTM")
    latest = inc[0] if inc else {}
    ni_fy, da_fy = latest.get("netIncome"), latest.get("depreciationAndAmortization")
    # FFO proxy (NAREIT FFO also adds back gains/losses on property sales — not available here).
    ffo = (ni_fy + da_fy) if (ni_fy is not None and da_fy is not None and not mismatch) else None
    eps_growth = None
    if len(inc) == 2 and inc[1].get("epsDiluted") not in (None, 0):
        eps_growth = (inc[0].get("epsDiluted", 0) - inc[1]["epsDiluted"]) / abs(inc[1]["epsDiluted"])
    return {
        "symbol": symbol,
        "company": profile.get("companyName"),
        "sector": profile.get("sector"),
        "industry": profile.get("industry"),
        "exchange": profile.get("exchange"),
        "price": profile.get("price"),
        "marketCap": mcap,
        "pe": pe,
        "peMeaningful": _meaningful(pe),
        # Earnings implied by cap / PE — keeps sign, so loss-makers enter aggregates.
        "earnings": (mcap / pe) if (mcap and pe not in (None, 0) and not mismatch) else None,
        "epsGrowth": eps_growth,
        "roe": km.get("returnOnEquityTTM"),
        "netMargin": ttm.get("netProfitMarginTTM"),
        "peg": None if mismatch else ttm.get("priceToEarningsGrowthRatioTTM"),
        "evToEbitda": None if mismatch else km.get("evToEBITDATTM"),
        "currencyMismatch": mismatch,
        "reportingCurrency": reporting,
        "kind": model.kind,
        "kindReason": model.reason,
        "pb": pb,
        # Book value implied by cap / P/B (TTM); FFO and FY net income in reporting currency.
        "book": (mcap / pb) if (mcap and pb and pb > 0) else None,
        "ffo": ffo,
        "netIncomeFY": None if mismatch else ni_fy,
    }


def _rows(symbols: list[str], client: FMPClient) -> pd.DataFrame:
    rows = [r for s in symbols if (r := _company_row(s, client)) is not None]
    return pd.DataFrame(rows)


def aggregate_pe(df: pd.DataFrame) -> float | None:
    """Σ market cap ÷ Σ earnings (index method). None if Σ earnings <= 0."""
    d = df.dropna(subset=["marketCap", "earnings"])
    total_e = d["earnings"].sum()
    return float(d["marketCap"].sum() / total_e) if total_e > 0 else None


def _premium(stock_pe: float | None, bench: float | None) -> float | None:
    if not _meaningful(stock_pe) or bench is None or bench <= 0:
        return None
    return float(stock_pe / bench - 1)  # type: ignore[operator]


# --------------------------------------------------------------------------- universes


def _similar_cap(rows: list[dict[str, Any]], cap: float | None) -> list[dict[str, Any]]:
    ref = cap or 1.0
    return sorted(rows, key=lambda r: abs(np.log((r.get("marketCap") or 1) / ref)))


def _peer_candidates(symbol: str, row: dict[str, Any], client: FMPClient) -> list[tuple[str, str]]:
    """Ordered (symbol, origin) candidates: same-industry US listings by cap
    similarity, then FMP stock-peers, then same-sector companies of similar
    size (0.2x-5x cap) as a fallback for thin industries (e.g. AAPL)."""
    cands: list[tuple[str, str]] = []
    major = ("NASDAQ", "NYSE", "AMEX")

    def screen(origin: str, **filters: Any) -> None:
        try:
            rows = client.screen_companies(limit=100, marketCapMoreThan=1_000_000_000, **filters)
        except FMPError as exc:
            logger.warning("Screener (%s) unavailable: %s", origin, exc)
            return
        rows = [
            r
            for r in rows
            if r.get("exchangeShortName") in major
            and not r.get("isEtf")
            and not r.get("isFund")
            and is_operating_equity(r)  # no preferreds / baby bonds / notes (EP-PC, KKRS, MER-PK)
        ]
        cands.extend((r["symbol"], origin) for r in _similar_cap(rows, row.get("marketCap")))

    screen("industry", industry=row.get("industry"))
    try:
        cands.extend((p["symbol"], "FMP peers") for p in client.get_stock_peers(symbol) if p.get("symbol"))
    except FMPError as exc:
        logger.warning("stock-peers unavailable for %s: %s", symbol, exc)
    cap = row.get("marketCap") or 0
    if cap:
        try:
            rows = client.screen_companies(
                limit=100,
                sector=row.get("sector"),
                marketCapMoreThan=int(cap * 0.2),
                marketCapLowerThan=int(cap * 5),
            )
            rows = [
                r
                for r in rows
                if r.get("exchangeShortName") in major
                and not r.get("isEtf")
                and not r.get("isFund")
                and is_operating_equity(r)
            ]
            cands.extend((r["symbol"], "sector, similar size") for r in _similar_cap(rows, cap))
        except FMPError as exc:
            logger.warning("Sector screener unavailable: %s", exc)
    seen: set[str] = set()
    out = []
    for sym, origin in cands:
        if sym != symbol and "." not in sym and sym not in seen:
            seen.add(sym)
            out.append((sym, origin))
    return out


def _select_peers(
    candidates: list[tuple[str, str]],
    row: dict[str, Any],
    client: FMPClient,
    max_peers: int,
    same_industry: bool = False,
    excluded_model: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Evaluate candidates in order until `max_peers` currency-consistent
    peers are found (at most 3x max_peers lookups). Returns (peers, excluded)."""
    chosen: list[dict[str, Any]] = []
    excluded: list[str] = []
    names: set[str] = {row.get("company") or ""}
    for sym, origin in candidates[: max_peers * 3]:
        if len(chosen) >= max_peers:
            break
        r = _company_row(sym, client)
        if r is None:
            continue
        if r["currencyMismatch"]:
            excluded.append(sym)
            continue
        if r["company"] in names:  # share-class duplicate (GOOG/GOOGL)
            continue
        if not is_operating_equity({"companyName": r["company"]}):  # FMP stock-peers aren't pre-screened
            continue
        # Same business model only: a card lender is not re-rated against V/MA (F3).
        if r["kind"] != row.get("kind"):
            if excluded_model is not None:
                excluded_model.append(f"{sym} ({r['kind']})")
            continue
        # FMP stock-peers span industries (JPM -> GS); the "Industry" set must stay one scope.
        if same_industry and r.get("industry") != row.get("industry"):
            continue
        names.add(r["company"])
        r["origin"] = origin
        chosen.append(r)
    return pd.DataFrame(chosen), excluded


def market_proxy(client: FMPClient, size: int = RELVAL_MARKET_PROXY_SIZE) -> pd.DataFrame:
    """Largest US-listed companies (NASDAQ/NYSE, US-domiciled) with TTM P/E.
    Cached as a whole for the 'market' TTL, since ~size*4 calls build it.
    """
    key = {"size": size, "minCap": RELVAL_MARKET_PROXY_MIN_CAP, "v": 1}
    cached = cache_get("relval_market", key, ttl_for("market"))
    if not is_miss(cached):
        return pd.DataFrame(cached)
    screened = client.screen_companies(limit=200, marketCapMoreThan=RELVAL_MARKET_PROXY_MIN_CAP, country="US")
    screened = [
        r
        for r in screened
        if r.get("exchangeShortName") in ("NASDAQ", "NYSE") and not r.get("isEtf") and not r.get("isFund")
    ]
    # Share-class duplicates (GOOGL/GOOG, BRK-A/BRK-B): keep the larger-volume line.
    by_name: dict[str, dict[str, Any]] = {}
    for r in screened:
        name = r.get("companyName") or r["symbol"]
        if name not in by_name or (r.get("volume") or 0) > (by_name[name].get("volume") or 0):
            by_name[name] = r
    top = sorted(by_name.values(), key=lambda r: r.get("marketCap") or 0, reverse=True)[:size]
    df = _rows([r["symbol"] for r in top], client)
    cache_set("relval_market", key, df.to_dict(orient="records"))
    return df


def _snapshot(client: FMPClient, kind: str, name: str, exchange: str) -> tuple[float | None, str | None]:
    """Most recent FMP sector/industry P/E (looks back up to 7 days for a trading day)."""
    fetch = client.get_industry_pe_snapshot if kind == "industry" else client.get_sector_pe_snapshot
    field_name = "industry" if kind == "industry" else "sector"
    for back in range(0, 8):
        d = (date.today() - timedelta(days=back)).isoformat()
        try:
            rows = fetch(d, exchange=exchange)
        except FMPError as exc:
            logger.warning("%s P/E snapshot unavailable: %s", kind, exc)
            return None, None
        match = [r for r in rows if r.get(field_name) == name and r.get("pe") is not None]
        if match:
            return float(match[0]["pe"]), d
    return None, None


# --------------------------------------------------------------------------- history


def _pe_history(symbol: str, client: FMPClient, years: int) -> pd.Series:
    """Daily TTM P/E = close ÷ Σ last-4-quarter diluted EPS, EPS known only
    from each quarter's filing date (point-in-time, no look-ahead)."""
    quarters = client.get_income_statement(symbol, period="quarter", limit=4 * years + 8)
    q = pd.DataFrame(
        [
            {
                "known": pd.Timestamp(r.get("filingDate") or r["date"]),
                "periodEnd": pd.Timestamp(r["date"]),
                "eps": r.get("epsDiluted"),
            }
            for r in quarters
            if r.get("epsDiluted") is not None
        ]
    ).sort_values("periodEnd")
    if len(q) < 4:
        return pd.Series(dtype=float)
    q["ttmEps"] = q["eps"].rolling(4).sum()
    q = q.dropna(subset=["ttmEps"]).sort_values("known")
    start = (date.today() - timedelta(days=365 * years)).isoformat()
    prices = client.get_historical_prices(symbol, start=start)
    px = pd.Series({pd.Timestamp(p["date"]): p["price"] for p in prices}).sort_index()
    eps = q.set_index("known")["ttmEps"]
    eps = eps[~eps.index.duplicated(keep="last")]
    eps_daily = eps.reindex(px.index.union(eps.index)).ffill().reindex(px.index)
    pe = px / eps_daily
    return pe.where(pe > 0).dropna()


def _agg_history(rows: list[dict[str, Any]]) -> pd.Series:
    if not rows:
        return pd.Series(dtype=float)
    s = pd.Series({pd.Timestamp(r["date"]): r.get("pe") for r in rows}).sort_index()
    return s.dropna().astype(float)


# --------------------------------------------------------------------------- main


def run_relative_valuation(
    symbol: str,
    max_peers: int = RELVAL_MAX_PEERS,
    history_years: int = RELVAL_HISTORY_YEARS,
    include_market: bool = True,
    client: FMPClient | None = None,
) -> RelativeValuationReport:
    symbol = normalize_symbol(symbol)
    if max_peers < 1:
        raise ValueError(f"max_peers must be >= 1, got {max_peers}")
    if not 1 <= history_years <= 10:
        raise ValueError(f"history_years must be within 1-10, got {history_years}")
    client = client or FMPClient()

    me = _company_row(symbol, client)
    if me is None:
        raise ValueError(f"No data available for {symbol}")
    if me["currencyMismatch"]:
        raise ValueError(
            f"{symbol} quotes in a different currency than it reports ({me['reportingCurrency']}); "
            "P/E comparisons would mix currencies and require FX conversion (not implemented)."
        )
    caveats: list[str] = []
    exchange = me["exchange"] if me["exchange"] in _SNAPSHOT_EXCHANGES else "NASDAQ"
    if me["exchange"] not in _SNAPSHOT_EXCHANGES:
        caveats.append(f"{symbol} lists on {me['exchange']}; FMP sector/industry P/E taken from NASDAQ.")

    # Peers.
    # Two peer sets, never blended: industry peers (primary) and size-matched
    # sector peers (broader reference; becomes primary only if the industry
    # yields fewer than RELVAL_MIN_MEANINGFUL_PEERS meaningful P/Es).
    cands = _peer_candidates(symbol, me, client)
    ind_cands = [c for c in cands if c[1] != "sector, similar size"]
    sec_cands = [c for c in cands if c[1] == "sector, similar size"]
    model_excluded: list[str] = []
    peers, mismatched = _select_peers(
        ind_cands, me, client, max_peers, same_industry=True, excluded_model=model_excluded
    )
    taken = set(peers["symbol"]) if not peers.empty else set()
    sector_peers, mism2 = _select_peers(
        [c for c in sec_cands if c[0] not in taken],
        me,
        client,
        min(RELVAL_SECTOR_PEERS, max_peers),
        excluded_model=model_excluded,
    )
    if model_excluded:
        caveats.append(
            f"Candidates excluded for a different business model than {symbol} ({me['kind']}): "
            + ", ".join(dict.fromkeys(model_excluded))
        )
    mismatched += mism2
    if mismatched:
        caveats.append(
            "Candidates excluded for quote/reporting currency mismatch: "
            + ", ".join(dict.fromkeys(mismatched))
        )
    n_ind_meaningful = int(peers["peMeaningful"].sum()) if not peers.empty else 0
    primary_is_sector = n_ind_meaningful < RELVAL_MIN_MEANINGFUL_PEERS and not sector_peers.empty
    if primary_is_sector:
        caveats.append(
            f"Only {n_ind_meaningful} industry peer(s) have a meaningful P/E (minimum {RELVAL_MIN_MEANINGFUL_PEERS}); "
            "the verdict uses the size-matched sector peer set instead — a broader, less precise comparison."
        )
    if not peers.empty:
        peers["set"] = "Industry"
    if not sector_peers.empty:
        sector_peers["set"] = "Sector (size-matched)"
    benchmarks: list[Benchmark] = []
    stock_pe = me["pe"]
    for label, df in (("Industry peers", peers), ("Sector peers (size-matched)", sector_peers)):
        if df.empty:
            continue
        mf = df[df["peMeaningful"]]
        nm = len(df) - len(mf)
        if nm:
            caveats.append(
                f"{label}: {nm} of {len(df)} with negative or > {RELVAL_MAX_MEANINGFUL_PE:.0f}× P/E ('not meaningful') — "
                "excluded from the median, included in the aggregate."
            )
        med = float(mf["pe"].median()) if not mf.empty else None
        agg = aggregate_pe(df)
        benchmarks.append(
            Benchmark(
                f"{label} — median", med, _premium(stock_pe, med), "median of meaningful TTM P/Es", len(mf)
            )
        )
        benchmarks.append(
            Benchmark(
                f"{label} — aggregate", agg, _premium(stock_pe, agg), "Σ market cap ÷ Σ earnings", len(df)
            )
        )
    primary_name = "Sector peers (size-matched) — median" if primary_is_sector else "Industry peers — median"
    primary_df = sector_peers if primary_is_sector else peers
    meaningful = primary_df[primary_df["peMeaningful"]] if not primary_df.empty else primary_df
    all_peers = (
        pd.concat([d for d in (peers, sector_peers) if not d.empty], ignore_index=True)
        if not (peers.empty and sector_peers.empty)
        else pd.DataFrame()
    )

    ind_pe, ind_date = _snapshot(client, "industry", me["industry"], exchange)
    benchmarks.append(
        Benchmark(
            f"Industry — {me['industry']}",
            ind_pe,
            _premium(stock_pe, ind_pe),
            f"FMP industry P/E, {exchange}, {ind_date}",
        )
    )
    sec_pe, sec_date = _snapshot(client, "sector", me["sector"], exchange)
    benchmarks.append(
        Benchmark(
            f"Sector — {me['sector']}",
            sec_pe,
            _premium(stock_pe, sec_pe),
            f"FMP sector P/E, {exchange}, {sec_date}",
        )
    )

    market = pd.DataFrame()
    if include_market:
        try:
            market = market_proxy(client)
            mkt_pe = aggregate_pe(market)
            mkt_med = float(market.loc[market["peMeaningful"], "pe"].median()) if not market.empty else None
            benchmarks.append(
                Benchmark(
                    "US large-cap market",
                    mkt_pe,
                    _premium(stock_pe, mkt_pe),
                    f"Σ cap ÷ Σ earnings, top {len(market)} US companies",
                    len(market),
                )
            )
            benchmarks.append(
                Benchmark(
                    "US large-cap median",
                    mkt_med,
                    _premium(stock_pe, mkt_med),
                    "median TTM P/E, same universe",
                    len(market),
                )
            )
        except FMPError as exc:
            caveats.append(f"Market proxy unavailable: {exc}")

    # History.
    history = pd.DataFrame()
    hstats: dict[str, Any] = {}
    try:
        start = (date.today() - timedelta(days=365 * history_years)).isoformat()
        stock_hist = _pe_history(symbol, client, history_years)
        ind_hist = _agg_history(
            client.get_historical_industry_pe(me["industry"], start, date.today().isoformat(), exchange)
        )
        sec_hist = _agg_history(
            client.get_historical_sector_pe(me["sector"], start, date.today().isoformat(), exchange)
        )
        history = pd.DataFrame({"stockPE": stock_hist, "industryPE": ind_hist, "sectorPE": sec_hist}).dropna(
            subset=["stockPE"]
        )
        history = history.ffill()
        history["relativeToIndustry"] = history["stockPE"] / history["industryPE"]
        history["relativeToSector"] = history["stockPE"] / history["sectorPE"]
        if len(history) > 60:
            s = history["stockPE"]
            rel = history["relativeToSector"].dropna()
            hstats = {
                "mean": float(s.mean()),
                "std": float(s.std(ddof=1)),
                "current": float(s.iloc[-1]),
                "z": float((s.iloc[-1] - s.mean()) / s.std(ddof=1)) if s.std(ddof=1) > 0 else None,
                "percentile": float((s <= s.iloc[-1]).mean() * 100),
                "min": float(s.min()),
                "max": float(s.max()),
                "relSectorMean": float(rel.mean()) if not rel.empty else None,
                "relSectorCurrent": float(rel.iloc[-1]) if not rel.empty else None,
                "relSectorZ": float((rel.iloc[-1] - rel.mean()) / rel.std(ddof=1))
                if len(rel) > 2 and rel.std(ddof=1) > 0
                else None,
                "start": s.index[0].date().isoformat(),
            }
        elif stock_hist.empty:
            caveats.append("P/E history unavailable (insufficient positive quarterly EPS).")
    except FMPError as exc:
        caveats.append(f"P/E history unavailable: {exc}")

    verdict, detail = _verdict(stock_pe, benchmarks, hstats, primary_name)
    rep = RelativeValuationReport(
        symbol=symbol,
        company=me["company"],
        sector=me["sector"],
        industry=me["industry"],
        exchange=me["exchange"],
        price=me["price"],
        stock=me,
        peers=all_peers,
        benchmarks=benchmarks,
        market_constituents=market,
        history=history,
        history_stats=hstats,
        verdict=verdict,
        verdict_detail=detail,
        caveats=caveats,
    )
    rep.primary_benchmark = primary_name
    _findings(rep, meaningful, ind_pe)
    rep.methodology = _methodology(rep, exchange)
    return rep


def _verdict(
    stock_pe: float | None, benches: list[Benchmark], h: dict[str, Any], primary: str
) -> tuple[str, str]:
    """Relative-valuation verdict from peer-median premium (primary — most
    comparable) and own-history z-score (secondary). Not a Buy/Sell rating."""
    if not _meaningful(stock_pe):
        return "Not meaningful", "TTM P/E is negative or extreme — use EV/Sales or EV/EBITDA instead."
    peer = next((b for b in benches if b.name == primary and b.premium is not None), None)
    z = h.get("z")
    votes = []
    if peer is not None:
        votes.append(1 if peer.premium > 0.15 else -1 if peer.premium < -0.15 else 0)  # type: ignore[operator]
    if z is not None:
        votes.append(1 if z > 1 else -1 if z < -1 else 0)
    if not votes:
        return "Insufficient data", "No meaningful peer or history benchmark."
    score = sum(votes) / len(votes)
    label = "Premium" if score >= 0.5 else "Discount" if score <= -0.5 else "In line"
    parts = []
    if peer is not None:
        parts.append(f"{peer.premium:+.0%} vs {peer.name.replace(' — median', '').lower()} median")
    if z is not None:
        parts.append(f"{z:+.1f}σ vs own {h.get('start', '')[:4]}+ history")
    return label, "; ".join(parts)


def _findings(rep: RelativeValuationReport, meaningful: pd.DataFrame, ind_pe: float | None) -> None:
    me = rep.stock
    pe = me["pe"]
    if _meaningful(pe):
        rep.findings.append(f"**{rep.symbol}** trades at **{pe:.1f}× TTM earnings**.")
    for b in rep.benchmarks:
        if b.pe is not None and b.premium is not None:
            rep.findings.append(
                f"vs {b.name}: {b.pe:.1f}× → **{b.premium:+.0%}** {'premium' if b.premium > 0 else 'discount'}."
            )
    h = rep.history_stats
    if h.get("z") is not None:
        rep.findings.append(
            f"Own history since {h['start'][:4]}: average {h['mean']:.1f}× (range {h['min']:.1f}–{h['max']:.1f}×); "
            f"today is the {h['percentile']:.0f}th percentile (z = {h['z']:+.1f})."
        )
    if h.get("relSectorZ") is not None:
        rep.findings.append(
            f"Relative to its sector, P/E is {h['relSectorCurrent']:.2f}× the sector multiple vs a "
            f"{h['relSectorMean']:.2f}× average (z = {h['relSectorZ']:+.1f})."
        )
    # Is the premium earned? Compare growth and ROE with peer medians.
    if not meaningful.empty:
        g_peer = meaningful["epsGrowth"].median()
        roe_peer = meaningful["roe"].median()
        g, roe = me.get("epsGrowth"), me.get("roe")
        if g is not None and pd.notna(g_peer) and roe is not None and pd.notna(roe_peer):
            better = (g > g_peer) + (roe > roe_peer)
            peer_prem = next((b.premium for b in rep.benchmarks if b.name == rep.primary_benchmark), None)
            justified = (
                "consistent with"
                if (peer_prem or 0) > 0 and better == 2
                else "not supported by"
                if (peer_prem or 0) > 0 and better == 0
                else "partly supported by"
                if (peer_prem or 0) > 0
                else "despite"
                if better == 2
                else "in line with"
            )
            rep.findings.append(
                f"Fundamentals: EPS growth {g:+.0%} vs peer median {g_peer:+.0%}; ROE {roe:.0%} vs {roe_peer:.0%} — the "
                f"valuation gap is {justified} relative fundamentals."
            )
    if ind_pe is not None and not meaningful.empty:
        med = meaningful["pe"].median()
        if med > 0 and (ind_pe / med > 2 or ind_pe / med < 0.5):
            rep.caveats.append(
                f"FMP's industry P/E ({ind_pe:.0f}×) diverges sharply from the peer median ({med:.0f}×) — the FMP "
                "aggregate is likely dominated by one or two mega-caps or near-zero earners; prefer the peer median."
            )


def _methodology(rep: RelativeValuationReport, exchange: str) -> list[str]:
    return [
        "**Multiple** — trailing-twelve-month P/E (price ÷ TTM diluted EPS), FMP ratios-ttm.",
        f"**Peer groups** — (1) Industry: up to {RELVAL_MAX_PEERS} currency-consistent same-industry US listings "
        "(market cap ≥ $1B, closest in size) plus FMP stock-peers; (2) Sector: up to "
        f"{RELVAL_SECTOR_PEERS} same-sector companies at 0.2×–5× the subject's market cap. The two are never "
        f"blended; the sector set becomes primary only when fewer than {RELVAL_MIN_MEANINGFUL_PEERS} industry peers "
        "have a meaningful P/E.",
        f"**Not meaningful (NM)** — P/E ≤ 0 or > {RELVAL_MAX_MEANINGFUL_PE:.0f}× excluded from medians; included in "
        "aggregates through their (negative/small) earnings, so aggregates reflect the true earnings pool.",
        "**Aggregate P/E** — Σ market cap ÷ Σ earnings (index construction). Avoids the upward bias of averaging ratios.",
        f"**Industry / sector** — FMP P/E snapshots for {exchange} listings (methodology not published by FMP; "
        "cross-checked against the peer median).",
        f"**Market** — US large-cap proxy: the {RELVAL_MARKET_PROXY_SIZE} largest US-domiciled NASDAQ/NYSE companies "
        "(share-class de-duplicated), aggregate and median P/E. Approximates the S&P 500 top-weight cohort.",
        "**History** — daily P/E from close ÷ TTM EPS (sum of 4 quarters), each quarter's EPS used only from its SEC "
        "filing date (point-in-time). Relative P/E = stock ÷ FMP sector P/E; z-score vs the window mean.",
        f"**Verdict** — Premium / In line / Discount from (a) premium to the primary peer median ({rep.primary_benchmark}) "
        "beyond ±15% and (b) own-history "
        "z-score beyond ±1σ. This is a relative-valuation reading, not an investment rating.",
        "**Currency guard** — quote vs reporting currency checked for the subject and every peer; mismatches are "
        "excluded (subject) or nulled (peers).",
    ]


# =========================================================================== peer valuation (price target input)
#
# One peer engine for the research report and the scorecard (replaces the
# legacy FMP-stock-peers average). Business-model-appropriate multiple
# (P/E, P/B, P/FFO — config.PEER_METHOD_BY_KIND), one peer scope at a time
# (industry → explicit sector fallback, never blended), same business model
# only, Σ numerator ÷ Σ denominator aggregate with the median alongside, and
# a minimum peer count below which no implied price is produced.

_METHODS: dict[str, dict[str, str]] = {
    "pe": {"label": "P/E (TTM)", "den": "earnings", "den_label": "earnings"},
    "pb": {"label": "P/B (TTM)", "den": "book", "den_label": "book value"},
    "pffo": {"label": "P/FFO (latest FY, FFO ≈ net income + D&A)", "den": "ffo", "den_label": "FFO"},
}


@dataclass
class PeerValuation:
    symbol: str
    kind: str
    method: str
    method_label: str
    price: float | None
    subject_multiple: float | None = None
    peer_aggregate: float | None = None
    peer_median: float | None = None
    n: int = 0
    scope: str | None = None
    implied_price: float | None = None
    peers: pd.DataFrame = field(default_factory=pd.DataFrame)
    excluded_fx: list[str] = field(default_factory=list)
    excluded_model: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    unavailable_reason: str | None = None
    subject_roe: float | None = None
    peer_roe: float | None = None
    # P/B only: justified-P/B scaling (ROE_s - g) / (ROE_p - g) applied to the peer aggregate
    roe_adjustment: float | None = None
    target_multiple: float | None = None
    # False when the implied price rests on an assumption flagged in notes; the report
    # shows it but never blends it into a price target.
    reliable: bool = True


def _usable(df: pd.DataFrame, den: str) -> pd.DataFrame:
    if df.empty or den not in df.columns:
        return pd.DataFrame()
    return df.dropna(subset=["marketCap", den])


def _multiple_stats(df: pd.DataFrame, den: str, method: str) -> tuple[float | None, float | None]:
    """(Σ cap ÷ Σ denominator, median of positive per-company multiples)."""
    total = float(df[den].sum())
    agg = float(df["marketCap"].sum()) / total if total > 0 else None
    mult = df["marketCap"] / df[den]
    mult = mult[mult > 0]
    if method == "pe":
        mult = mult[mult <= RELVAL_MAX_MEANINGFUL_PE]
    return agg, (float(mult.median()) if not mult.empty else None)


def peer_valuation(
    symbol: str, client: FMPClient | None = None, max_peers: int = RELVAL_MAX_PEERS
) -> PeerValuation:
    """Peer-multiple implied price for `symbol` using the multiple that fits its business model."""
    symbol = normalize_symbol(symbol)
    if max_peers < 1:
        raise ValueError(f"max_peers must be >= 1, got {max_peers}")
    client = client or FMPClient()
    me = _company_row(symbol, client)
    if me is None:
        raise ValueError(f"No profile data for {symbol}")
    method = PEER_METHOD_BY_KIND[me["kind"]]
    spec = _METHODS[method]
    den = spec["den"]
    pv = PeerValuation(symbol, me["kind"], method, spec["label"], me.get("price"))
    if me["currencyMismatch"]:
        pv.unavailable_reason = (
            f"{symbol} quotes in a different currency than it reports ({me['reportingCurrency']}) — "
            "price multiples would mix currencies (FX conversion not implemented)"
        )
        return pv

    mcap, sub_den = me.get("marketCap"), me.get(den)
    pv.subject_multiple = float(mcap / sub_den) if (mcap and sub_den and sub_den > 0) else None

    cands = _peer_candidates(symbol, me, client)
    ind_cands = [c for c in cands if c[1] != "sector, similar size"]
    sec_cands = [c for c in cands if c[1] == "sector, similar size"]
    chosen = pd.DataFrame()
    for scope, pool, same_ind in (
        ("Industry peers", ind_cands, True),
        ("Sector peers (size-matched)", sec_cands, False),
    ):
        df, fx = _select_peers(
            pool, me, client, max_peers, same_industry=same_ind, excluded_model=pv.excluded_model
        )
        pv.excluded_fx += fx
        usable = _usable(df, den)
        if len(usable) >= PEER_TARGET_MIN_PEERS:
            chosen, pv.scope = usable, scope
            break
        pv.notes.append(
            f"{scope}: only {len(usable)} usable peer(s) with {spec['den_label']} "
            f"(minimum {PEER_TARGET_MIN_PEERS})."
        )
    pv.excluded_fx = list(dict.fromkeys(pv.excluded_fx))
    pv.excluded_model = list(dict.fromkeys(pv.excluded_model))
    if chosen.empty:
        pv.unavailable_reason = (
            f"fewer than {PEER_TARGET_MIN_PEERS} same-business-model peers with {spec['den_label']} "
            "in either the industry or the size-matched sector set"
        )
        return pv

    pv.peers, pv.n = chosen.reset_index(drop=True), len(chosen)
    pv.peer_aggregate, pv.peer_median = _multiple_stats(chosen, den, method)
    if method == "pe" and pv.peer_aggregate is not None and pv.peer_aggregate > RELVAL_MAX_MEANINGFUL_PE:
        pv.notes.append(
            f"Peer aggregate P/E of {pv.peer_aggregate:,.0f}× (Σ earnings near zero — loss-makers offset "
            f"profits) is not meaningful (> {RELVAL_MAX_MEANINGFUL_PE:.0f}×)."
        )
        pv.peer_aggregate = None

    if method == "pb":
        pv.subject_roe = me.get("roe")
        with_e = chosen.dropna(subset=["earnings", "book"])
        tb = float(with_e["book"].sum()) if not with_e.empty else 0.0
        pv.peer_roe = float(with_e["earnings"].sum()) / tb if tb > 0 else None
        g = DCF_TERMINAL_GROWTH_RATE
        sr, pr = pv.subject_roe, pv.peer_roe
        if sr is not None and pr is not None and sr > g and pr > g:
            # Gordon justified P/B = (ROE - g) / (r - g); relative to peers the cost of equity
            # cancels, so the subject's warranted P/B = peer P/B x (ROE_s - g) / (ROE_p - g).
            pv.roe_adjustment = (sr - g) / (pr - g)
            pv.method_label = "P/B (TTM), ROE-adjusted"
            pv.notes.append(
                f"P/B scaled for profitability: ROE {sr:.1%} vs peer aggregate {pr:.1%} → warranted P/B = "
                f"peer P/B × ({sr:.1%} − {g:.1%}) ÷ ({pr:.1%} − {g:.1%}) = × {pv.roe_adjustment:.2f} "
                "(assumes similar cost of equity and long-run growth)."
            )
        else:
            pv.reliable = False
            pv.notes.append(
                "P/B not ROE-adjusted (subject or peer ROE missing or below long-run growth) — the plain P/B "
                "re-rating is shown for reference only."
            )
            if sr is not None and pr is not None and abs(sr - pr) * 100 > PEER_ROE_GAP_CAVEAT_PP:
                pv.notes.append(
                    f"ROE {sr:.1%} vs peer aggregate {pr:.1%}: P/B gap is largely a profitability gap."
                )
    if method == "pe":
        ttm_e, fy_e = me.get("earnings"), me.get("netIncomeFY")
        if ttm_e and fy_e and ttm_e > 0 and fy_e > 0:
            ratio = max(ttm_e, fy_e) / min(ttm_e, fy_e)
            if ratio > PEER_EPS_TTM_DIVERGENCE:
                pv.notes.append(
                    f"TTM earnings and latest-FY earnings differ {ratio:.1f}× (one-off year?) — the TTM P/E "
                    "used here may not reflect normalised earnings."
                )
    if method == "pffo":
        pv.notes.append(
            "FFO ≈ net income + D&A (gains on property sales are not removed — NAREIT FFO would be lower "
            "in years with large disposals)."
        )

    if pv.subject_multiple is None:
        pv.unavailable_reason = f"{symbol}'s own {spec['den_label']} is non-positive — no meaningful multiple"
    elif pv.peer_aggregate is None:
        pv.unavailable_reason = f"peer Σ {spec['den_label']} is non-positive"
    elif pv.price:
        pv.target_multiple = pv.peer_aggregate * (pv.roe_adjustment or 1.0)
        pv.implied_price = float(pv.price * pv.target_multiple / pv.subject_multiple)
    return pv
