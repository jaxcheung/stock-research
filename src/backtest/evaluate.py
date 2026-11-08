# src/backtest/evaluate.py
#
# Honest performance evaluation for monthly backtests.
#   - Standard metrics: CAGR, volatility, Sharpe and Sortino (on excess returns over T-bills),
#     max drawdown, Calmar, turnover.
#   - Versus the benchmark: alpha and beta from an excess-return regression with Newey-West
#     (HAC) standard errors (timeseries.ols_hac) — alpha reported with t-stat, p-value and n;
#     tracking error and information ratio.
#   - Overfitting controls:
#       * Deflated Sharpe ratio (Bailey & Lopez de Prado, 2014) applied to ACTIVE returns vs the
#         benchmark (i.e. a deflated information ratio): probability the strategy truly beats the
#         benchmark after allowing for the number of strategies ever tried (persistent ledger,
#         trials.py), their dispersion, and non-normal returns. On raw excess returns it would
#         only measure the equity premium — SPY itself scored 94%.
#       * Stationary block bootstrap confidence interval for the Sharpe and for the active return.
#       * Walk-forward: the best strategy is SELECTED on the in-sample period and judged on the
#         out-of-sample period it never saw.
# Nothing here decides a strategy is "good": the report states plainly when it is not.

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.analysis import timeseries as ts
from src.backtest.engine import BacktestResult
from src.config import (
    BACKTEST_BOOTSTRAP_BLOCK,
    BACKTEST_BOOTSTRAP_SAMPLES,
    BACKTEST_MIN_EVAL_MONTHS,
    BACKTEST_WALK_FORWARD_TEST_FRACTION,
    STATS_SIGNIFICANCE_LEVEL,
)

PPY = 12  # monthly


def max_drawdown(r: pd.Series) -> float:
    wealth = (1 + r).cumprod()
    return float((wealth / wealth.cummax() - 1).min()) if not r.empty else float("nan")


def metrics(r: pd.Series, rf: pd.Series, turnover: pd.Series | None = None) -> dict[str, Any]:
    r = r.dropna()
    if len(r) < 12:
        return {"n": len(r)}
    ex = r - rf.reindex(r.index).fillna(0.0)
    years = len(r) / PPY
    cagr = float(np.prod(1 + r.to_numpy(dtype=float)) ** (1 / years) - 1)
    vol = float(r.std(ddof=1) * np.sqrt(PPY))
    sharpe = float(ex.mean() / ex.std(ddof=1) * np.sqrt(PPY)) if ex.std(ddof=1) > 0 else float("nan")
    # Downside deviation: root-mean-square of min(excess, 0) over ALL months (Sortino & Price),
    # not the RMS of the negative months only — that overstates downside risk and shrinks Sortino.
    dd = float(np.sqrt(np.mean(np.minimum(ex.to_numpy(dtype=float), 0.0) ** 2)) * np.sqrt(PPY))
    sortino = float(ex.mean() * PPY / dd) if dd > 0 else float("nan")
    mdd = max_drawdown(r)
    return {
        "n": len(r),
        "start": r.index.min().strftime("%Y-%m"),
        "end": r.index.max().strftime("%Y-%m"),
        "cagr": cagr,
        "volatility": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "maxDrawdown": mdd,
        "calmar": cagr / abs(mdd) if mdd < 0 else float("nan"),
        "hitRate": float((r > 0).mean()),
        "skew": float(stats.skew(ex)),
        "kurtosis": float(stats.kurtosis(ex, fisher=False)),
        "avgTurnover": float(turnover.reindex(r.index).mean()) if turnover is not None else float("nan"),
    }


def vs_benchmark(r: pd.Series, bench: pd.Series, rf: pd.Series) -> dict[str, Any]:
    """Alpha/beta (HAC), tracking error, information ratio, and active-return bootstrap CI."""
    df = pd.concat([r.rename("p"), bench.rename("b")], axis=1).dropna()
    if len(df) < 24:
        return {}
    rfa = rf.reindex(df.index).fillna(0.0)
    reg = ts.ols_hac(df["p"] - rfa, (df["b"] - rfa).to_frame("benchmark"))
    c = reg["coefficients"]
    active = df["p"] - df["b"]
    te = float(active.std(ddof=1) * np.sqrt(PPY))
    lo, hi = bootstrap_ci(active, lambda x: float(np.mean(x)) * PPY)
    return {
        "alphaAnnual": float(c.loc["const", "coef"]) * PPY,
        "alphaT": float(c.loc["const", "t"]),
        "alphaP": float(c.loc["const", "p"]),
        "beta": float(c.loc["benchmark", "coef"]),
        "trackingError": te,
        "informationRatio": float(active.mean() * PPY / te) if te > 0 else float("nan"),
        "activeReturnAnnual": float(active.mean() * PPY),
        "activeCI95": (lo, hi),
        "n": int(reg["n"]),
        "hacLags": int(reg["hacLags"]),
    }


def bootstrap_ci(
    x: pd.Series, stat: Any, block: int = BACKTEST_BOOTSTRAP_BLOCK, samples: int = BACKTEST_BOOTSTRAP_SAMPLES
) -> tuple[float, float]:
    """95% CI of stat(x) by stationary block bootstrap (Politis & Romano), fixed seed."""
    v = x.dropna().to_numpy(dtype=float)
    n = len(v)
    if n < 2 * block:
        return float("nan"), float("nan")
    rng = np.random.default_rng(12345)
    p = 1.0 / block
    out = np.empty(samples)
    for k in range(samples):
        idx = np.empty(n, dtype=int)
        idx[0] = rng.integers(n)
        for i in range(1, n):
            idx[i] = rng.integers(n) if rng.random() < p else (idx[i - 1] + 1) % n
        out[k] = stat(v[idx])
    lo, hi = np.nanpercentile(out, [2.5, 97.5])
    return float(lo), float(hi)


def sharpe_annual(x: np.ndarray) -> float:
    sd = float(np.std(x, ddof=1))
    return float(np.mean(x) / sd * np.sqrt(PPY)) if sd > 0 else float("nan")


def deflated_sharpe(
    sr_annual: float, n_obs: int, skew: float, kurt: float, trial_srs: list[float]
) -> dict[str, Any]:
    """Deflated Sharpe ratio. sr_annual: the candidate's annualised Sharpe; trial_srs: annualised
    Sharpes of ALL strategies tried (including the candidate). Returns the probability that the
    true Sharpe exceeds the best one expected by luck among that many trials."""
    trials = [s for s in trial_srs if np.isfinite(s)]
    n_trials = len(trials)
    if n_obs < 24 or n_trials < 1 or not np.isfinite(sr_annual):
        return {"dsr": float("nan"), "nTrials": n_trials}
    sr = sr_annual / np.sqrt(PPY)  # per-period
    var_trials = float(np.var(np.asarray(trials) / np.sqrt(PPY), ddof=1)) if n_trials > 1 else 0.0
    gamma = 0.5772156649
    if n_trials > 1 and var_trials > 0:
        z1 = stats.norm.ppf(1 - 1 / n_trials)
        z2 = stats.norm.ppf(1 - 1 / (n_trials * np.e))
        sr0 = np.sqrt(var_trials) * ((1 - gamma) * z1 + gamma * z2)
    else:
        sr0 = 0.0
    denom = np.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr**2))
    dsr = float(stats.norm.cdf((sr - sr0) * np.sqrt(n_obs - 1) / denom))
    return {"dsr": dsr, "nTrials": n_trials, "expectedMaxSharpeByLuck": float(sr0 * np.sqrt(PPY))}


@dataclass
class EvaluationReport:
    table: pd.DataFrame  # one row per strategy: metrics + vs benchmark + DSR
    walk_forward: pd.DataFrame  # per strategy: in-sample vs out-of-sample Sharpe / alpha
    selection: dict[str, Any]  # best-in-sample strategy and how it did out of sample
    findings: list[str] = field(default_factory=list)
    methodology: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)


def evaluate(
    results: dict[str, BacktestResult],
    rf: pd.Series,
    benchmark: str = "spy",
    controls: tuple[str, ...] = (),
    test_fraction: float = BACKTEST_WALK_FORWARD_TEST_FRACTION,
    prior_trial_irs: list[float] | None = None,
) -> EvaluationReport:
    """prior_trial_irs: information ratios of OTHER configurations tried before (trial ledger);
    they count towards the deflated-Sharpe trial number and dispersion."""
    if benchmark not in results:
        raise ValueError(f"benchmark result {benchmark!r} missing")
    if not 0.1 <= test_fraction <= 0.5:
        raise ValueError("test_fraction must be between 0.1 and 0.5")
    bench = results[benchmark].returns
    if bench.dropna().size < BACKTEST_MIN_EVAL_MONTHS:
        raise ValueError(
            f"Evaluation needs at least {BACKTEST_MIN_EVAL_MONTHS} months (got {bench.dropna().size}); "
            "widen the window"
        )
    rows = []
    for name, res in results.items():
        m = metrics(res.returns, rf, res.turnover)
        vb = vs_benchmark(res.returns, bench, rf) if name != benchmark else {}
        sr_lo, sr_hi = bootstrap_ci(res.returns - rf.reindex(res.returns.index).fillna(0.0), sharpe_annual)
        # Second, separately-labelled test for stock strategies: does the selection beat the
        # stocks it chooses FROM (equal-weight eligible universe)? Answers "does the signal pick
        # better stocks" independently of the universe vs SPY (equal weight vs cap weight).
        vu: dict[str, Any] = {}
        if res.kind == "stock" and "universe_ew" in results and name != "universe_ew":
            u = vs_benchmark(res.returns, results["universe_ew"].returns, rf)
            vu = {"alphaVsUniverse": u.get("alphaAnnual"), "alphaPVsUniverse": u.get("alphaP")}
        rows.append(
            {
                "strategy": name,
                "kind": res.kind,
                **vu,
                "control": name in controls,
                **m,
                **vb,
                "sharpeCI95": (sr_lo, sr_hi),
            }
        )
    table = pd.DataFrame(rows).set_index("strategy")
    trial_names = [n for n in table.index if not table.loc[n, "control"] and n != benchmark]
    # Deflated Sharpe on ACTIVE returns (strategy − benchmark): "does it beat the benchmark after
    # multiple testing", not "does it beat T-bills" (which any equity portfolio does).
    active_stats: dict[str, tuple[float, int, float, float]] = {}
    for n in trial_names:
        a = (results[n].returns - bench).dropna()
        if len(a) < BACKTEST_MIN_EVAL_MONTHS:
            continue
        active_stats[n] = (
            sharpe_annual(a.to_numpy(dtype=float)),
            len(a),
            float(stats.skew(a)),
            float(stats.kurtosis(a, fisher=False)),
        )
    trial_irs = [v[0] for v in active_stats.values()] + list(prior_trial_irs or [])
    for n, (ir, n_obs, sk, ku) in active_stats.items():
        d = deflated_sharpe(ir, n_obs, sk, ku, trial_irs)
        table.loc[n, "deflatedSharpe"] = d["dsr"]
        table.loc[n, "nTrials"] = d["nTrials"]
        table.loc[n, "expectedMaxIRByLuck"] = d.get("expectedMaxSharpeByLuck", np.nan)

    # Walk-forward: split the common window by time; select on IS, judge on OOS.
    common = bench.index
    split_at = common[int(len(common) * (1 - test_fraction))]
    wf_rows = []
    for name, res in results.items():
        r = res.returns
        is_r, oos_r = r[r.index < split_at], r[r.index >= split_at]
        mi, mo = metrics(is_r, rf), metrics(oos_r, rf)
        vi = vs_benchmark(is_r, bench, rf) if name != benchmark else {}
        vo = vs_benchmark(oos_r, bench, rf) if name != benchmark else {}
        wf_rows.append(
            {
                "strategy": name,
                "startIS": mi.get("start"),
                "endIS": mi.get("end"),
                "startOOS": mo.get("start"),
                "endOOS": mo.get("end"),
                "sharpeIS": mi.get("sharpe"),
                "sharpeOOS": mo.get("sharpe"),
                "alphaIS": vi.get("alphaAnnual"),
                "alphaOOS": vo.get("alphaAnnual"),
                "alphaPOOS": vo.get("alphaP"),
                "irOOS": vo.get("informationRatio"),
            }
        )
    wf = pd.DataFrame(wf_rows).set_index("strategy")
    cand = wf.loc[trial_names].dropna(subset=["sharpeIS"]) if trial_names else wf.iloc[0:0]
    selection: dict[str, Any] = {"splitAt": split_at.strftime("%Y-%m")}
    if not cand.empty:
        best = str(cand["sharpeIS"].idxmax())
        selection.update(
            {
                "best": best,
                "sharpeIS": _num(cand.at[best, "sharpeIS"]),
                "sharpeOOS": _num(cand.at[best, "sharpeOOS"]),
                "benchSharpeOOS": _num(wf.at[benchmark, "sharpeOOS"]),
                "alphaOOS": _num(cand.at[best, "alphaOOS"]),
                "alphaPOOS": _num(cand.at[best, "alphaPOOS"]),
                "nOOS": int(len(bench[bench.index >= split_at])),
            }
        )
    rep = EvaluationReport(table, wf, selection)
    rep.findings = _findings(rep, benchmark, trial_names, len(trial_irs))
    rep.methodology = _methodology(benchmark, test_fraction)
    rep.caveats = _caveats(results)
    return rep


def _num(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _fmt(v: Any, f: str) -> str:
    return format(v, f) if isinstance(v, int | float) and np.isfinite(v) else "n/a"


def _findings(rep: EvaluationReport, benchmark: str, trials: list[str], n_trials: int) -> list[str]:
    t, out = rep.table, []
    b = t.loc[benchmark]
    out.append(
        f"Benchmark ({benchmark.upper()}): CAGR {_fmt(b['cagr'], '.1%')}, Sharpe {_fmt(b['sharpe'], '.2f')}, "
        f"max drawdown {_fmt(b['maxDrawdown'], '.0%')} ({b['start']}–{b['end']}, n = {int(b['n'])} months)."
    )
    alpha = t.loc[trials].dropna(subset=["alphaP"]) if trials else t.iloc[0:0]
    sig = alpha[(alpha["alphaP"] < STATS_SIGNIFICANCE_LEVEL) & (alpha["alphaAnnual"] > 0)]
    if sig.empty:
        out.append(
            f"No strategy shows statistically significant positive alpha vs {benchmark.upper()} at the "
            f"{STATS_SIGNIFICANCE_LEVEL:.0%} level (HAC t-tests) — none can be said to beat the market on this data."
        )
    for name, r in sig.iterrows():
        dsr = r.get("deflatedSharpe")
        out.append(
            f"**{name}**: alpha {_fmt(r['alphaAnnual'], '+.1%')}/yr (t = {_fmt(r['alphaT'], '.2f')}, "
            f"p = {_fmt(r['alphaP'], '.3f')}, n = {int(r['n'])}), deflated-Sharpe probability of beating "
            f"{benchmark.upper()} {_fmt(dsr, '.0%')} given {n_trials} configurations ever tried"
            + (
                " — does NOT survive the multiple-testing adjustment."
                if isinstance(dsr, float) and dsr < 0.95
                else "."
            )
        )
    if "alphaPVsUniverse" in t.columns:
        vu = t.loc[[n for n in trials if n in t.index]].dropna(subset=["alphaPVsUniverse"])
        for name, r in vu.iterrows():
            a, p = _num(r["alphaVsUniverse"]), _num(r["alphaPVsUniverse"])
            if p < STATS_SIGNIFICANCE_LEVEL:
                out.append(
                    f"**{name}** vs the equal-weight universe it picks from: alpha {_fmt(a, '+.1%')}/yr "
                    f"(p = {_fmt(p, '.3f')}) — "
                    + (
                        "it selects better-than-average stocks"
                        if a > 0
                        else "it selects worse-than-average stocks"
                    )
                    + ", a separate question from beating SPY (not adjusted for multiple testing)."
                )
    sel = rep.selection
    if "best" in sel:
        oos_beats = _num(sel["sharpeOOS"]) > _num(sel["benchSharpeOOS"])
        p_oos = _num(sel.get("alphaPOOS"))
        significant = p_oos < STATS_SIGNIFICANCE_LEVEL and _num(sel.get("alphaOOS")) > 0
        untestable = not np.isfinite(p_oos)
        verdict = (
            "had a higher Sharpe out of sample, but the window is too short to test alpha "
            f"(needs {BACKTEST_MIN_EVAL_MONTHS} months)."
            if untestable and oos_beats
            else "did NOT beat the benchmark out of sample."
            if not oos_beats
            else "beat it out of sample with significant alpha."
            if significant
            else "had a higher Sharpe out of sample, but its alpha is NOT statistically significant — "
            "not evidence of an edge."
        )
        out.append(
            f"Walk-forward: picking the best in-sample strategy (**{sel['best']}**, Sharpe {_fmt(sel['sharpeIS'], '.2f')} "
            f"before {sel['splitAt']}) gave an out-of-sample Sharpe of {_fmt(sel['sharpeOOS'], '.2f')} vs the "
            f"benchmark's {_fmt(sel['benchSharpeOOS'], '.2f')}; "
            + (
                f"OOS n = {sel.get('nOOS')} months"
                if untestable
                else f"OOS alpha {_fmt(sel.get('alphaOOS'), '+.1%')}/yr (p = {_fmt(p_oos, '.2f')}, "
                f"n = {sel.get('nOOS')} months)"
            )
            + " — it "
            + verdict
        )
    return out


def _methodology(benchmark: str, test_fraction: float) -> list[str]:
    return [
        "Monthly rebalancing at month-end close; long-only; returns are dividend-adjusted total returns earned the "
        "FOLLOWING month; transaction costs = turnover × cost bps; uninvested weight earns the 3-month T-bill.",
        "Point-in-time: fundamentals usable only from max(filing/acceptance date, fiscal year end + 60 days) + 1 day; "
        "universe membership uses market cap = month-end close × shares from the latest filed statement, plus a "
        "liquidity screen (median daily dollar volume over the last 63 sessions ≥ $5m).",
        "Sortino uses downside deviation over all months (target = T-bill). "
        f"Sharpe/Sortino on excess returns over T-bills, annualised ×√12. Alpha/beta vs {benchmark.upper()} from an "
        "excess-return regression with Newey-West (HAC) standard errors; alpha shown with t, p and n.",
        f"Deflated Sharpe (Bailey & López de Prado 2014) on ACTIVE returns vs {benchmark.upper()} (a deflated "
        "information ratio): probability the strategy truly beats the benchmark after allowing for the best result "
        "expected from luck across EVERY configuration ever tried (persistent trial ledger — re-running one "
        "strategy at a time does not reset the count) and for skew/kurtosis. Controls are not counted as trials.",
        "Placebo: many random-pick portfolios (different seeds), gross of costs, regressed on the equal-weight "
        "universe. About 5% should look 'significant' by chance; materially more flags a biased pipeline.",
        "95% confidence intervals by stationary block bootstrap (6-month blocks, 2,000 resamples, fixed seed).",
        f"Walk-forward: the last {test_fraction:.0%} of months are held out; the strategy with the best in-sample "
        "Sharpe is selected and judged only on the held-out months.",
    ]


def _caveats(results: dict[str, BacktestResult]) -> list[str]:
    out = [
        "Survivorship bias: the stock universe is built from companies listed today; companies that were delisted, "
        "acquired or went bankrupt are absent, which flatters stock strategies (ETF results are less affected).",
        "Sector labels are today's FMP classification, applied to all history.",
        "Annual fundamentals only (Starter tier): signals update once a year per company, lagging quarterly data.",
        "Fundamentals are FMP's current (restated) figures, not as originally filed; the filing date gates WHEN a "
        "year is used, but restatements made later are visible to the backtest (mild look-ahead).",
        "Value signals (E/P, B/P) use only companies reporting in USD — price is in USD, so non-USD reporters "
        "(Canadian, UK, Chinese filers) would need FX conversion (currency rule).",
        "Past performance — even out of sample — does not guarantee future results. Research only, not advice.",
    ]
    miss = sum(r.missing_returns for r in results.values())
    if miss:
        out.append(
            f"{miss} position-months across strategies had no next-month return and were assumed flat (data gaps / "
            "stopped trading)."
        )
    return out
