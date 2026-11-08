# src/backtest/lab.py
#
# Orchestrates a backtest suite: load point-in-time data once, run every requested strategy
# (plus the benchmark and controls), and evaluate them together so the deflated Sharpe and
# the walk-forward selection see every strategy that was tried.

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.analysis import timeseries as ts
from src.backtest import trials
from src.backtest.data import BacktestData, load_backtest_data
from src.backtest.engine import BacktestResult, run_backtest
from src.backtest.evaluate import EvaluationReport, evaluate, sharpe_annual, vs_benchmark
from src.backtest.strategies import CONTROLS, NON_TRIALS, STRATEGIES, make_placebo, universe_at
from src.config import (
    BACKTEST_BENCHMARK,
    BACKTEST_COST_BPS,
    BACKTEST_MIN_EVAL_MONTHS,
    BACKTEST_MIN_UNIVERSE,
    BACKTEST_PLACEBO_SEEDS,
    BACKTEST_SCORECARD_MIN_GROWTH_COVERAGE,
    BACKTEST_TOP_N,
    SCORECARD_RULES_VERSION,
    STATS_SIGNIFICANCE_LEVEL,
)
from src.utils.logging import get_logger

logger = get_logger("backtest.lab")


@dataclass
class LabReport:
    results: dict[str, BacktestResult]
    evaluation: EvaluationReport
    data_notes: list[str] = field(default_factory=list)
    placebo: dict[str, Any] = field(default_factory=dict)

    def equity_curves(self) -> pd.DataFrame:
        return pd.DataFrame({n: (1 + r.returns).cumprod() for n, r in self.results.items()})


def run_lab(
    strategies: list[str],
    start: str | None = "2011-01-31",
    end: str | None = None,
    cost_bps: float = BACKTEST_COST_BPS,
    data: BacktestData | None = None,
    include_controls: bool = True,
    placebo_seeds: int = BACKTEST_PLACEBO_SEEDS,
    record_trials: bool = True,
    rebalance_every: int = 1,
    top_n: int | None = None,
    keep_weights: bool = False,
) -> LabReport:
    """rebalance_every / top_n apply to every strategy in the suite (and are part of each trial's
    identity in the ledger); keep_weights retains holdings for the portfolio simulator."""
    unknown = [s for s in strategies if s not in STRATEGIES]
    if unknown:
        raise ValueError(f"Unknown strategies {unknown}; choose from {sorted(STRATEGIES)}")
    data = data or load_backtest_data()
    bench = BACKTEST_BENCHMARK.lower()
    kinds = {STRATEGIES[n].kind for n in strategies}
    # Controls only for the kinds being tested: a stock control in an ETF-only suite would pull the
    # whole window back to the first full stock universe (~2017).
    controls = [c for c in CONTROLS if include_controls and STRATEGIES[c].kind in kinds | {"etf"}]
    names = list(dict.fromkeys([bench, *strategies, *controls]))
    if "universe_ew" not in names and placebo_seeds and "stock" in kinds:
        names.append("universe_ew")  # the placebo's reference portfolio
    notes = [n for n in data.notes if not n.startswith("Universe source")]  # shown once, as a caveat
    if any(STRATEGIES[n].kind == "stock" for n in names):
        # FMP Starter serves 10 annual statements, so the point-in-time stock universe only exists
        # from ~2017. Every strategy — ETFs and the benchmark too — is evaluated on the SAME window,
        # starting at the first month with a full stock universe; before this fix, stock strategies
        # sat in T-bills for 2011-16 and "lost" to SPY by construction.
        first = first_full_universe(data, start)
        if first is not None and (start is None or first > pd.Timestamp(start)):
            notes.append(
                f"Stock universe has ≥ {BACKTEST_MIN_UNIVERSE} eligible companies only from {first:%Y-%m} (annual "
                "statements on FMP Starter go back 10 years), so the whole suite starts there. Run ETF strategies "
                "alone for the longer history."
            )
            start = first.strftime("%Y-%m-%d")
    if any(n.startswith("scorecard") for n in names):
        from src.backtest.scorecard import first_full_scorecard

        full = first_full_scorecard(data, start)
        if full is not None and (start is None or full > pd.Timestamp(start)):
            notes.append(
                f"The fundamental scorecard's growth dimension needs two filed full years, so it covers "
                f"≥ {BACKTEST_SCORECARD_MIN_GROWTH_COVERAGE:.0%} of the universe only from {full:%Y-%m}; the suite "
                "starts there (earlier months would test a different, 3-dimension model)."
            )
            start = full.strftime("%Y-%m-%d")
    results: dict[str, BacktestResult] = {}
    for n in names:
        logger.info("Backtesting %s", n)
        results[n] = run_backtest(
            STRATEGIES[n],
            data,
            start=start,
            end=end,
            cost_bps=cost_bps,
            top_n=top_n,
            keep_weights=keep_weights,
            rebalance_every=rebalance_every,
        )
    params: dict[str, Any] = {"start": start, "end": end, "costBps": cost_bps, "benchmark": bench}
    if rebalance_every != 1:
        params["rebalanceEvery"] = rebalance_every  # omitted at the default: earlier trial keys stay valid
    if top_n is not None and top_n != BACKTEST_TOP_N:
        params["topN"] = top_n  # the default is omitted: the same configuration is ONE trial
    tried = [n for n in results if n not in NON_TRIALS and n != bench]

    def trial_params(n: str) -> dict[str, Any]:
        # A scorecard RULE revision is a new trial (rules changed after seeing results must count).
        return {**params, "scorecardRules": SCORECARD_RULES_VERSION} if n.startswith("scorecard") else params

    ledger = trials.load()
    keys = {trials.trial_key(n, trial_params(n)) for n in tried}
    prior = [
        float(v)
        for k, v in zip(ledger["key"], ledger["info_ratio"], strict=True)
        if k not in keys and pd.notna(v)
    ]
    ev = evaluate(
        results,
        data.rf,
        benchmark=bench,
        controls=tuple(c for c in (*CONTROLS, "placebo_random") if c in results),
        prior_trial_irs=prior,
        rebalance_every=rebalance_every,
    )
    if record_trials:
        b = results[bench].returns
        for n in tried:
            a = (results[n].returns - b).dropna()
            if len(a) >= BACKTEST_MIN_EVAL_MONTHS:
                trials.record(n, trial_params(n), sharpe_annual(a.to_numpy(dtype=float)), len(a))
    if rebalance_every > 1:
        _anchor_range(ev, results, data, tried, bench, start, end, cost_bps, top_n, rebalance_every)
    placebo: dict[str, Any] = {}
    if placebo_seeds and "universe_ew" in results:
        placebo = run_placebo(
            data,
            results["universe_ew"],
            placebo_seeds,
            start,
            end,
            rebalance_every=rebalance_every,
            top_n=top_n,
        )
        ev.findings.append(placebo["finding"])
    ev.caveats.extend(n for n in data.notes if n.startswith("Universe source"))
    for r in results.values():
        ev.caveats.extend(f"{r.strategy}: {n}" for n in r.notes)
    return LabReport(results, ev, notes, placebo)


def _anchor_range(
    ev: EvaluationReport,
    results: dict[str, BacktestResult],
    data: BacktestData,
    tried: list[str],
    bench: str,
    start: str | None,
    end: str | None,
    cost_bps: float,
    top_n: int | None,
    k: int,
) -> None:
    """With a k-month cadence, WHICH months trade is a free choice that moves results by several
    points. Re-run every tried strategy at all k anchors (same window) and report the spread, so
    the default anchor is never presented as the result when it is just one draw."""
    b = results[bench].returns
    rows: list[str] = []
    for n in tried:
        cagrs, alphas = [], []
        for a in range(k):
            r = (
                results[n]
                if a == 0
                else run_backtest(
                    STRATEGIES[n],
                    data,
                    start=start,
                    end=end,
                    cost_bps=cost_bps,
                    top_n=top_n,
                    rebalance_every=k,
                    anchor=a,
                )
            ).returns.dropna()
            cagrs.append(float(np.prod(1 + r.to_numpy(dtype=float)) ** (12 / len(r)) - 1))
            vb = vs_benchmark(r, b, data.rf)
            alphas.append(float(vb.get("alphaAnnual", np.nan)))
        c = np.asarray(cagrs)
        rank = int((c > c[0]).sum()) + 1  # rank of the default anchor, 1 = best
        ev.table.loc[n, "cagrAnchorMin"] = float(c.min())
        ev.table.loc[n, "cagrAnchorMedian"] = float(np.median(c))
        ev.table.loc[n, "cagrAnchorMax"] = float(c.max())
        ev.table.loc[n, "alphaAnchorMedian"] = float(np.nanmedian(alphas))
        rows.append(
            f"{n} {c.min():.1%}–{c.max():.1%} (median {np.median(c):.1%}; the default anchor ranks {rank} of {k})"
        )
    if rows:
        ev.findings.append(
            f"Rebalance-anchor sensitivity (every {k} months — CAGR across all {k} possible trading months, same "
            "window): " + "; ".join(rows) + ". Judge the median, not the default anchor."
        )


def run_placebo(
    data: BacktestData,
    universe_ew: BacktestResult,
    seeds: int,
    start: str | None,
    end: str | None,
    rebalance_every: int = 1,
    top_n: int | None = None,
) -> dict[str, Any]:
    """Random-pick portfolios over many seeds, GROSS of costs, each regressed on the equal-weight
    universe's gross return (HAC). With an unbiased pipeline about a `STATS_SIGNIFICANCE_LEVEL` share of seeds look significant
    (either sign) by chance; far more means the pipeline leaks information."""
    ref = universe_ew.gross
    rf = data.rf
    rows = []
    for seed in range(seeds):
        r = run_backtest(
            make_placebo(seed),
            data,
            start=start,
            end=end,
            cost_bps=0.0,
            rebalance_every=rebalance_every,  # same cadence as the suite, so drift/hold is tested too
            top_n=top_n,
        ).gross
        df = pd.concat([r.rename("p"), ref.rename("u")], axis=1).dropna()
        if len(df) < BACKTEST_MIN_EVAL_MONTHS:
            continue
        rfa = rf.reindex(df.index).fillna(0.0)
        reg = ts.ols_hac(df["p"] - rfa, (df["u"] - rfa).to_frame("universe"))
        c = reg["coefficients"]
        rows.append(
            {
                "seed": seed,
                "alphaAnnual": float(c.loc["const", "coef"]) * 12,
                "alphaP": float(c.loc["const", "p"]),
            }
        )
    tab = pd.DataFrame(rows)
    if tab.empty:
        return {"table": tab, "finding": "Placebo: not enough months to test."}
    frac = float((tab["alphaP"] < STATS_SIGNIFICANCE_LEVEL).mean())
    # Binomial tail: chance of at least this many 'significant' seeds if the pipeline is unbiased.
    k = int((tab["alphaP"] < STATS_SIGNIFICANCE_LEVEL).sum())
    p_binom = float(stats.binom.sf(k - 1, len(tab), STATS_SIGNIFICANCE_LEVEL)) if k else 1.0
    biased = p_binom < 0.01
    borderline = not biased and p_binom < 0.05
    finding = (
        f"Placebo: {len(tab)} random-pick portfolios (gross of costs) vs the equal-weight universe — median alpha "
        f"{np.median(tab['alphaAnnual']):+.1%}/yr, {k} of {len(tab)} ({frac:.0%}) significant at "
        f"{STATS_SIGNIFICANCE_LEVEL:.0%} (≈{STATS_SIGNIFICANCE_LEVEL:.0%} expected by chance; binomial p = {p_binom:.2f}) — "
        + (
            "MORE than chance: the weighting/return mechanics are biased; do not trust other results."
            if biased
            else "BORDERLINE — slightly more than chance; the HAC t-test over-rejects at this sample size (about "
            "7–9% of placebos look significant at 5%), so this is weak evidence of bias. This checks"
            if borderline
            else "consistent with chance. This checks the engine's weighting and return mechanics only: random "
            "picks and the reference share one universe, so it CANNOT detect survivorship or selection look-ahead."
        )
    )
    return {
        "table": tab,
        "significantShare": frac,
        "binomialP": p_binom,
        "biased": biased,
        "finding": finding,
    }


def first_full_universe(data: BacktestData, start: str | None = None) -> pd.Timestamp | None:
    """First month-end (>= start) whose point-in-time stock universe has >= BACKTEST_MIN_UNIVERSE names."""
    lo = pd.Timestamp(start) if start else None
    for t in pd.DatetimeIndex(data.returns.index):
        if lo is not None and t < lo:
            continue
        if len(universe_at(data, t)[0]) >= BACKTEST_MIN_UNIVERSE:
            return t
    return None
