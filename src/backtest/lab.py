# src/backtest/lab.py
#
# Orchestrates a backtest suite: load point-in-time data once, run every requested strategy
# (plus the benchmark and controls), and evaluate them together so the deflated Sharpe and
# the walk-forward selection see every strategy that was tried.

from dataclasses import dataclass, field

import pandas as pd

from src.backtest.data import BacktestData, load_backtest_data
from src.backtest.engine import BacktestResult, run_backtest
from src.backtest.evaluate import EvaluationReport, evaluate
from src.backtest.strategies import CONTROLS, STRATEGIES, universe_at
from src.config import BACKTEST_BENCHMARK, BACKTEST_COST_BPS, BACKTEST_MIN_UNIVERSE
from src.utils.logging import get_logger

logger = get_logger("backtest.lab")


@dataclass
class LabReport:
    results: dict[str, BacktestResult]
    evaluation: EvaluationReport
    data_notes: list[str] = field(default_factory=list)

    def equity_curves(self) -> pd.DataFrame:
        return pd.DataFrame({n: (1 + r.returns).cumprod() for n, r in self.results.items()})


def run_lab(
    strategies: list[str],
    start: str | None = "2011-01-31",
    end: str | None = None,
    cost_bps: float = BACKTEST_COST_BPS,
    data: BacktestData | None = None,
    include_controls: bool = True,
) -> LabReport:
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
    notes = list(data.notes)
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
    results: dict[str, BacktestResult] = {}
    for n in names:
        logger.info("Backtesting %s", n)
        results[n] = run_backtest(STRATEGIES[n], data, start=start, end=end, cost_bps=cost_bps)
    ev = evaluate(results, data.rf, benchmark=bench, controls=tuple(c for c in CONTROLS if c in results))
    for r in results.values():
        ev.caveats.extend(f"{r.strategy}: {n}" for n in r.notes)
    return LabReport(results, ev, notes)


def first_full_universe(data: BacktestData, start: str | None = None) -> pd.Timestamp | None:
    """First month-end (>= start) whose point-in-time stock universe has >= BACKTEST_MIN_UNIVERSE names."""
    lo = pd.Timestamp(start) if start else None
    for t in pd.DatetimeIndex(data.returns.index):
        if lo is not None and t < lo:
            continue
        if len(universe_at(data, t)[0]) >= BACKTEST_MIN_UNIVERSE:
            return t
    return None
