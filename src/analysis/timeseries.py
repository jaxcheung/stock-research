# src/analysis/timeseries.py
#
# Pure statistical engine for multi-variable time-series research. No I/O:
# every function takes pandas objects and returns pandas objects / dicts, so
# it works the same for macro series, prices, P/E histories or company data.
#
# Methodology choices (each surfaced in report "Methodology" sections):
# - Correlation of LEVELS between trending series is usually spurious
#   (Granger & Newbold 1974). Every series is ADF-tested; correlations are
#   reported on both levels and stationary transforms, and the engine
#   recommends which to trust.
# - Regression standard errors are Newey-West HAC, since macro residuals are
#   autocorrelated and plain OLS t-stats overstate significance.
# - Cycles: Hodrick-Prescott trend/cycle split (lambda by Ravn-Uhlig), then
#   a periodogram restricted to the business-cycle band (1.5-10y, Burns-
#   Mitchell). Phase follows the OECD business-cycle-clock convention.
# - Forecasts are always scored out-of-sample against a naive (random-walk)
#   benchmark; a model that can't beat naive is reported as such.

from typing import Any

import numpy as np
import pandas as pd
from scipy import signal, sparse, stats
from scipy.sparse.linalg import spsolve

from src.config import (
    CYCLE_MAX_PERIOD_YEARS,
    CYCLE_MIN_PERIOD_YEARS,
    FORECAST_BACKTEST_FRACTION,
    HP_LAMBDA,
    STATS_MAX_LAG,
    STATS_MIN_OBSERVATIONS,
    STATS_ROLLING_WINDOW,
    STATS_SIGNIFICANCE_LEVEL,
)

FREQ_ORDER = {"D": 0, "W": 1, "M": 2, "Q": 3, "A": 4}
PERIODS_PER_YEAR = {"D": 252, "W": 52, "M": 12, "Q": 4, "A": 1}
_PANDAS_FREQ = {"W": "W-FRI", "M": "ME", "Q": "QE", "A": "YE"}

TRANSFORMS = {
    "level": "Level",
    "diff": "Period change (Δ)",
    "pct": "Period % change",
    "yoy": "Year-over-year % change",
    "yoy_diff": "Year-over-year change (Δ, pp)",
    "log": "Log level",
    "zscore": "Z-score (standardised level)",
}
# Differencing transforms — the only ones that can remove a unit root. "log" and
# "zscore" are monotone rescalings of the level and inherit its non-stationarity.
STATIONARY_TRANSFORMS = ("diff", "pct", "yoy", "yoy_diff")


# --------------------------------------------------------------------------- alignment


def coarsest_freq(freqs: list[str]) -> str:
    return max(freqs, key=lambda f: FREQ_ORDER.get(f, 2)) if freqs else "M"


def align(
    series: dict[str, pd.Series],
    native_freqs: dict[str, str],
    freq: str = "auto",
    how: str = "mean",
    start: str | None = None,
    end: str | None = None,
) -> tuple[pd.DataFrame, str, list[str]]:
    """Resample each series to a common frequency and inner-join on dates.

    freq="auto" picks the coarsest native frequency, so nothing is upsampled
    (upsampling an annual series to monthly invents 11 data points a year).
    If a finer target is forced, coarser series are step-filled and a note
    is returned. `how` is the aggregation for downsampling ("mean" suits
    rates/flows; "last" suits stocks/levels like prices).
    """
    if not series:
        raise ValueError("series must be non-empty")
    if how not in ("mean", "last"):
        raise ValueError(f"how must be 'mean' or 'last', got {how!r}")
    notes: list[str] = []
    target = coarsest_freq(list(native_freqs.values())) if freq == "auto" else freq
    if freq == "auto" and target == "D":
        # Daily series don't share trading calendars (FRED vs exchange holidays);
        # weekly is the finest common grid the engine supports.
        target = "W"
        notes.append("All series are daily; aligned at weekly frequency (Friday, period mean).")
    if target not in _PANDAS_FREQ:
        raise ValueError(f"freq must be one of auto/W/M/Q/A, got {freq!r}")

    cols = {}
    for name, s in series.items():
        s = s.sort_index()
        if start:
            s = s.loc[pd.Timestamp(start) :]
        if end:
            s = s.loc[: pd.Timestamp(end)]
        resampled = getattr(s.resample(_PANDAS_FREQ[target]), how)()
        if FREQ_ORDER.get(native_freqs.get(name, "M"), 2) > FREQ_ORDER[target]:
            resampled = resampled.ffill(limit=PERIODS_PER_YEAR[target])
            notes.append(
                f"{name}: native frequency ({native_freqs[name]}) is coarser than {target}; "
                "values step-filled — treat its correlations with caution."
            )
        cols[name] = resampled
    df = pd.DataFrame(cols).dropna(how="any")
    if df.empty:
        raise ValueError("Series have no overlapping observations at this frequency/date range")
    return df, target, notes


def transform(df: pd.DataFrame, kind: str, freq: str) -> pd.DataFrame:
    """Apply a named transform (see TRANSFORMS) column-wise."""
    ppy = PERIODS_PER_YEAR[freq]
    if kind == "level":
        out = df.copy()
    elif kind == "diff":
        out = df.diff()
    elif kind == "pct":
        out = df.pct_change(fill_method=None) * 100
    elif kind == "yoy":
        out = df.pct_change(periods=ppy, fill_method=None) * 100
    elif kind == "yoy_diff":
        out = df.diff(periods=ppy)
    elif kind == "log":
        out = df.where(df > 0).apply(np.log)
    elif kind == "zscore":
        out = (df - df.mean()) / df.std(ddof=1)
    else:
        raise ValueError(f"Unknown transform {kind!r}; expected one of {list(TRANSFORMS)}")
    return out.replace([np.inf, -np.inf], np.nan).dropna(how="any")


def index_to_base(df: pd.DataFrame, base: float = 100.0) -> pd.DataFrame:
    """Rebase each column to `base` at its first observation (for one-axis comparison)."""
    first = df.iloc[0].replace(0, np.nan)
    return df.divide(first) * base


# --------------------------------------------------------------------------- stationarity


# MacKinnon (1996) asymptotic critical values, constant-only ADF regression.
_ADF_CRITICAL = {0.01: -3.43, 0.05: -2.86, 0.10: -2.57}


def adf_test(s: pd.Series) -> dict[str, Any]:
    """Augmented Dickey-Fuller test (constant, Schwert lag rule).
    H0: unit root (non-stationary). Reject when stat < critical value.
    """
    y = s.dropna().to_numpy(dtype=float)
    n = len(y)
    if n < STATS_MIN_OBSERVATIONS + 4:
        return {"stat": None, "stationary": None, "lags": None, "note": "too few observations"}
    lags = min(int(12 * (n / 100) ** 0.25), n // 4)
    dy = np.diff(y)
    rows = len(dy) - lags
    X = [np.ones(rows), y[lags:-1]]
    X += [dy[lags - i : len(dy) - i] for i in range(1, lags + 1)]
    Xm = np.column_stack(X)
    target = dy[lags:]
    coef, *_ = np.linalg.lstsq(Xm, target, rcond=None)
    resid = target - Xm @ coef
    dof = rows - Xm.shape[1]
    if dof <= 0:
        return {"stat": None, "stationary": None, "lags": lags, "note": "too few observations"}
    sigma2 = resid @ resid / dof
    try:
        cov = sigma2 * np.linalg.inv(Xm.T @ Xm)
    except np.linalg.LinAlgError:
        return {"stat": None, "stationary": None, "lags": lags, "note": "singular design"}
    stat = float(coef[1] / np.sqrt(cov[1, 1]))
    crit = _ADF_CRITICAL[0.05]
    return {"stat": stat, "critical5": crit, "stationary": stat < crit, "lags": lags, "note": ""}


# --------------------------------------------------------------------------- descriptive


def _trend_per_year(s: pd.Series, freq: str) -> tuple[float | None, float | None]:
    if len(s) < STATS_MIN_OBSERVATIONS:
        return None, None
    t = np.arange(len(s)) / PERIODS_PER_YEAR[freq]
    res = stats.linregress(t, s.to_numpy(dtype=float))
    return float(res.slope), float(res.pvalue)


def descriptive_stats(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Per-variable summary: distribution, where the latest value sits
    historically (percentile, z-score), trend, volatility, persistence, ADF.
    """
    ppy = PERIODS_PER_YEAR[freq]
    rows = []
    for col in df.columns:
        s = df[col].dropna()
        if s.empty:
            continue
        last = float(s.iloc[-1])
        std = float(s.std(ddof=1)) if len(s) > 1 else float("nan")
        slope, slope_p = _trend_per_year(s, freq)
        changes = s.diff().dropna()
        adf = adf_test(s)
        rows.append(
            {
                "variable": col,
                "n": len(s),
                "start": s.index[0].date().isoformat(),
                "end": s.index[-1].date().isoformat(),
                "last": last,
                "mean": float(s.mean()),
                "median": float(s.median()),
                "std": std,
                "min": float(s.min()),
                "max": float(s.max()),
                "lastPercentile": float(stats.percentileofscore(s, last, kind="weak")),
                "lastZScore": (last - float(s.mean())) / std if std and std > 0 else None,
                "change1y": last - float(s.iloc[-1 - ppy]) if len(s) > ppy else None,
                "trendPerYear": slope,
                "trendPValue": slope_p,
                "volatilityAnnualised": float(changes.std(ddof=1) * np.sqrt(ppy))
                if len(changes) > 1
                else None,
                "autocorrLag1": float(s.autocorr(lag=1)) if len(s) > 2 else None,
                "adfStat": adf["stat"],
                "stationary": adf["stationary"],
            }
        )
    return pd.DataFrame(rows).set_index("variable")


# --------------------------------------------------------------------------- correlation


def _corr_with_p(a: np.ndarray, b: np.ndarray, method: str) -> tuple[float, float]:
    if method == "spearman":
        r, p = stats.spearmanr(a, b)
    else:
        r, p = stats.pearsonr(a, b)
    return float(r), float(p)


def correlation_matrix(df: pd.DataFrame, method: str = "pearson") -> dict[str, pd.DataFrame]:
    """Pairwise correlation, p-value and n (pairwise-complete) for every column pair."""
    if method not in ("pearson", "spearman"):
        raise ValueError(f"method must be 'pearson' or 'spearman', got {method!r}")
    cols = list(df.columns)
    r = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
    p = pd.DataFrame(np.zeros((len(cols), len(cols))), index=cols, columns=cols)
    n = pd.DataFrame(np.zeros((len(cols), len(cols)), dtype=int), index=cols, columns=cols)
    for i, a in enumerate(cols):
        for j in range(i + 1, len(cols)):
            b = cols[j]
            pair = df[[a, b]].dropna()
            if len(pair) < STATS_MIN_OBSERVATIONS or pair[a].std() == 0 or pair[b].std() == 0:
                r.loc[a, b] = r.loc[b, a] = np.nan
                p.loc[a, b] = p.loc[b, a] = np.nan
            else:
                rv, pv = _corr_with_p(pair[a].to_numpy(), pair[b].to_numpy(), method)
                r.loc[a, b] = r.loc[b, a] = rv
                p.loc[a, b] = p.loc[b, a] = pv
            n.loc[a, b] = n.loc[b, a] = len(pair)
        n.loc[a, a] = int(df[a].notna().sum())
    return {"r": r, "p": p, "n": n}


def rolling_correlation(df: pd.DataFrame, a: str, b: str, window: int = STATS_ROLLING_WINDOW) -> pd.Series:
    if window < 3:
        raise ValueError(f"window must be >= 3, got {window}")
    return df[a].rolling(window).corr(df[b]).dropna()


def lead_lag(df: pd.DataFrame, a: str, b: str, max_lag: int = STATS_MAX_LAG) -> pd.DataFrame:
    """Cross-correlation corr(a_t, b_{t+k}) for k in [-max_lag, max_lag].
    Positive k with the peak |r| means `a` LEADS `b` by k periods.
    """
    if max_lag < 0:
        raise ValueError(f"max_lag must be >= 0, got {max_lag}")
    rows = []
    for k in range(-max_lag, max_lag + 1):
        pair = pd.concat([df[a], df[b].shift(-k)], axis=1).dropna()
        if len(pair) < STATS_MIN_OBSERVATIONS:
            continue
        r, p = stats.pearsonr(pair.iloc[:, 0], pair.iloc[:, 1])
        rows.append({"lag": k, "r": float(r), "p": float(p), "n": len(pair)})
    return pd.DataFrame(rows)


def best_lag(ll: pd.DataFrame) -> dict[str, Any] | None:
    if ll.empty:
        return None
    row = ll.loc[ll["r"].abs().idxmax()]
    return {"lag": int(row["lag"]), "r": float(row["r"]), "p": float(row["p"]), "n": int(row["n"])}


# --------------------------------------------------------------------------- regression


def ols_hac(y: pd.Series, X: pd.DataFrame) -> dict[str, Any]:
    """OLS with Newey-West (HAC) standard errors and an intercept.

    Bandwidth: floor(4 * (n/100)^(2/9)) — Newey & West (1994) rule of thumb.
    """
    data = pd.concat([y.rename("__y"), X], axis=1).dropna()
    n, k = len(data), X.shape[1] + 1
    if n < max(STATS_MIN_OBSERVATIONS, k + 3):
        raise ValueError(
            f"Need at least {max(STATS_MIN_OBSERVATIONS, k + 3)} overlapping observations, got {n}"
        )
    yv = data["__y"].to_numpy(dtype=float)
    Xv = np.column_stack([np.ones(n), data[X.columns].to_numpy(dtype=float)])
    xtx_inv = np.linalg.pinv(Xv.T @ Xv)
    beta = xtx_inv @ Xv.T @ yv
    resid = yv - Xv @ beta

    L = int(np.floor(4 * (n / 100) ** (2 / 9)))
    u = Xv * resid[:, None]
    S = u.T @ u
    for lag in range(1, L + 1):
        w = 1 - lag / (L + 1)
        G = u[lag:].T @ u[:-lag]
        S += w * (G + G.T)
    cov = xtx_inv @ S @ xtx_inv * n / (n - k)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, beta / se, np.nan)
    pvals = 2 * stats.t.sf(np.abs(t), df=n - k)

    ss_res = float(resid @ resid)
    ss_tot = float(((yv - yv.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    names = ["const", *X.columns]
    return {
        "coefficients": pd.DataFrame({"coef": beta, "seHAC": se, "t": t, "p": pvals}, index=names),
        "r2": r2,
        "adjR2": 1 - (1 - r2) * (n - 1) / (n - k) if n > k else float("nan"),
        "n": n,
        "durbinWatson": float(np.sum(np.diff(resid) ** 2) / ss_res) if ss_res > 0 else float("nan"),
        "residStd": float(np.sqrt(ss_res / (n - k))),
        "hacLags": L,
        "fitted": pd.Series(Xv @ beta, index=data.index),
    }


# --------------------------------------------------------------------------- cycles


def hp_filter(s: pd.Series, lamb: float) -> tuple[pd.Series, pd.Series]:
    """Hodrick-Prescott filter. Returns (trend, cycle)."""
    y = s.to_numpy(dtype=float)
    n = len(y)
    if n < 4:
        raise ValueError("HP filter needs at least 4 observations")
    D = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n))
    trend = spsolve((sparse.eye(n) + lamb * (D.T @ D)).tocsc(), y)
    trend_s = pd.Series(trend, index=s.index)
    return trend_s, s - trend_s


def _phase(cycle: pd.Series, ppy: int) -> str:
    """OECD business-cycle clock: sign of the gap × direction of the gap."""
    window = max(2, ppy // 4)
    recent = cycle.iloc[-(window + 1) :]
    above = recent.iloc[-1] > 0
    rising = recent.iloc[-1] > recent.iloc[0]
    if above and rising:
        return "Expansion"
    if above:
        return "Slowdown"
    if rising:
        return "Recovery"
    return "Contraction"


PHASE_DESCRIPTIONS = {
    "Expansion": "above trend and rising",
    "Slowdown": "above trend but falling — typically late-cycle",
    "Contraction": "below trend and falling",
    "Recovery": "below trend but rising — typically early-cycle",
}


def cycle_analysis(s: pd.Series, freq: str, log_scale: bool | None = None) -> dict[str, Any]:
    """Trend/cycle decomposition, dominant cycle length, turning points, phase.

    log_scale: decompose log(s) so the gap is "% from trend" — right for
    levels/indices (GDP, prices), wrong for rates already in percent (a rate
    moving 2%→3% is a 1 pp gap, not +50%). None = auto (log if all positive).
    """
    s = s.dropna()
    ppy = PERIODS_PER_YEAR[freq]
    if len(s) < max(STATS_MIN_OBSERVATIONS, int(CYCLE_MIN_PERIOD_YEARS * ppy * 2)):
        raise ValueError(
            f"Need at least {int(CYCLE_MIN_PERIOD_YEARS * ppy * 2)} observations for cycle analysis, got {len(s)}"
        )
    use_log = bool((s > 0).all()) if log_scale is None else (log_scale and bool((s > 0).all()))
    base = np.log(s) if use_log else s
    trend_b, cycle_b = hp_filter(base, HP_LAMBDA[freq])
    trend = np.exp(trend_b) if use_log else trend_b
    cycle = cycle_b * 100 if use_log else cycle_b  # % deviation from trend, or absolute gap

    # Periodogram restricted to the business-cycle band.
    freqs, power = signal.periodogram(cycle.to_numpy(dtype=float), fs=ppy, detrend="linear", window="hann")
    with np.errstate(divide="ignore"):
        periods = np.where(freqs > 0, 1 / freqs, np.inf)
    band = (periods >= CYCLE_MIN_PERIOD_YEARS) & (periods <= CYCLE_MAX_PERIOD_YEARS)
    total_power = float(power[freqs > 0].sum())
    dominant_period = None
    band_share = None
    if band.any() and total_power > 0:
        idx = np.argmax(np.where(band, power, -1))
        dominant_period = float(periods[idx])
        band_share = float(power[band].sum() / total_power)
    spectrum = pd.DataFrame({"periodYears": periods[freqs > 0], "power": power[freqs > 0]})

    # Turning points: peaks/troughs at least half a minimum cycle apart, and
    # prominent relative to the cycle's own volatility (filters noise wiggles).
    distance = max(1, int(CYCLE_MIN_PERIOD_YEARS * ppy / 2))
    prominence = 0.5 * float(cycle.std(ddof=1))
    peaks, _ = signal.find_peaks(cycle.to_numpy(), distance=distance, prominence=prominence)
    troughs, _ = signal.find_peaks(-cycle.to_numpy(), distance=distance, prominence=prominence)
    peak_dates = list(cycle.index[peaks])
    trough_dates = list(cycle.index[troughs])

    def _avg_gap_years(dates: list[pd.Timestamp]) -> float | None:
        if len(dates) < 2:
            return None
        return float(np.mean(np.diff([d.value for d in dates])) / (365.25 * 86400e9))

    avg_peak_to_peak = _avg_gap_years(peak_dates)
    last_peak = peak_dates[-1] if peak_dates else None
    last_trough = trough_dates[-1] if trough_dates else None
    phase = _phase(cycle, ppy)

    # Naive timing projection: next turning point = last turning point +
    # half the average full cycle. Reported as indicative only.
    next_turn = None
    half_cycle = (avg_peak_to_peak or dominant_period or 0) / 2
    if half_cycle and (last_peak or last_trough):
        last_turn = max(d for d in (last_peak, last_trough) if d is not None)
        kind = "trough" if last_turn == last_peak else "peak"
        next_turn = {"type": kind, "date": last_turn + pd.Timedelta(days=int(half_cycle * 365.25))}

    return {
        "trend": trend,
        "cycle": cycle,
        "cycleUnit": "% from trend" if use_log else "gap vs trend, in series units",
        "spectrum": spectrum,
        "dominantPeriodYears": dominant_period,
        "businessCycleBandShare": band_share,
        "peaks": peak_dates,
        "troughs": trough_dates,
        "avgPeakToPeakYears": avg_peak_to_peak,
        "phase": phase,
        "phaseDescription": PHASE_DESCRIPTIONS[phase],
        "currentGap": float(cycle.iloc[-1]),
        "gapPercentile": float(stats.percentileofscore(cycle, cycle.iloc[-1], kind="weak")),
        "nextTurningPoint": next_turn,
        "hpLambda": HP_LAMBDA[freq],
        "logTransformed": use_log,
    }


# --------------------------------------------------------------------------- forecasting


def _fit_harmonic(
    s: pd.Series, freq: str, horizon: int, log_scale: bool | None = None
) -> tuple[pd.Series, float]:
    """Local linear trend + dominant-cycle harmonic, fitted jointly by least squares.

    The dominant period comes from the HP cycle's periodogram; the regression
    y = a + b*t + c*sin(wt) + d*cos(wt) is then fitted on the most recent
    ~2.5 cycles (the trend slope is local, not whole-history). Fitting the
    trend jointly avoids the HP filter's end-point bias, which otherwise
    leaks the latest cycle swing into the extrapolated trend.
    Returns (forecast for `horizon` periods, in-sample residual std on the model scale).
    """
    ppy = PERIODS_PER_YEAR[freq]
    use_log = bool((s > 0).all()) if log_scale is None else (log_scale and bool((s > 0).all()))
    base = np.log(s) if use_log else s
    _, cycle = hp_filter(base, HP_LAMBDA[freq])

    freqs, power = signal.periodogram(cycle.to_numpy(dtype=float), fs=ppy, detrend="linear")
    with np.errstate(divide="ignore"):
        periods = np.where(freqs > 0, 1 / freqs, np.inf)
    band = (periods >= CYCLE_MIN_PERIOD_YEARS) & (periods <= CYCLE_MAX_PERIOD_YEARS)
    period_periods = float(periods[np.argmax(np.where(band, power, -1))] * ppy) if band.any() else None

    window = len(base) if period_periods is None else min(len(base), max(int(2.5 * period_periods), 2 * ppy))
    y = base.iloc[-window:].to_numpy(dtype=float)
    t = np.arange(window, dtype=float)
    t_future = np.arange(window, window + horizon, dtype=float)

    def design(tt: np.ndarray) -> np.ndarray:
        cols = [np.ones_like(tt), tt]
        if period_periods:
            w = 2 * np.pi / period_periods
            cols += [np.sin(w * tt), np.cos(w * tt)]
        return np.column_stack(cols)

    coef, *_ = np.linalg.lstsq(design(t), y, rcond=None)
    resid_std = float(np.std(y - design(t) @ coef, ddof=design(t).shape[1]))
    fc_base = design(t_future) @ coef

    idx = pd.date_range(s.index[-1], periods=horizon + 1, freq=_PANDAS_FREQ.get(freq, "ME"))[1:]
    return pd.Series(np.exp(fc_base) if use_log else fc_base, index=idx), resid_std


def forecast(s: pd.Series, freq: str, horizon: int, log_scale: bool | None = None) -> dict[str, Any]:
    """Cycle-aware forecast with an out-of-sample backtest against naive.

    Skill = 1 - RMSE(model)/RMSE(naive) on the held-out tail (model re-fit on
    the training window only — no look-ahead). Skill <= 0 means the model
    does not beat "tomorrow = today" and the forecast should not be relied on.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if freq == "D":
        raise ValueError("Resample daily data to W/M/Q/A before forecasting")
    s = s.dropna()
    ppy = PERIODS_PER_YEAR[freq]
    min_n = int(CYCLE_MIN_PERIOD_YEARS * ppy * 3)
    if len(s) < min_n:
        raise ValueError(f"Need at least {min_n} observations to forecast at {freq} frequency, got {len(s)}")

    holdout = max(horizon, int(len(s) * FORECAST_BACKTEST_FRACTION))
    holdout = min(holdout, len(s) - min_n // 2)
    train, test = s.iloc[:-holdout], s.iloc[-holdout:]
    bt_fc, _ = _fit_harmonic(train, freq, holdout, log_scale)
    bt_fc.index = test.index
    naive = pd.Series(train.iloc[-1], index=test.index)
    rmse_model = float(np.sqrt(((bt_fc - test) ** 2).mean()))
    rmse_naive = float(np.sqrt(((naive - test) ** 2).mean()))
    skill = 1 - rmse_model / rmse_naive if rmse_naive > 0 else None

    fc, resid_std = _fit_harmonic(s, freq, horizon, log_scale)
    # Prediction band: residual std widened with horizon (random-walk error growth).
    use_log = bool((s > 0).all()) if log_scale is None else (log_scale and bool((s > 0).all()))
    widen = resid_std * 1.96 * np.sqrt(np.arange(1, horizon + 1))
    if use_log:
        lower, upper = fc * np.exp(-widen), fc * np.exp(widen)
    else:
        lower, upper = fc - widen, fc + widen

    return {
        "forecast": fc,
        "lower95": pd.Series(lower, index=fc.index),
        "upper95": pd.Series(upper, index=fc.index),
        "backtest": pd.DataFrame({"actual": test, "model": bt_fc, "naive": naive}),
        "rmseModel": rmse_model,
        "rmseNaive": rmse_naive,
        "skill": skill,
        "beatsNaive": skill is not None and skill > 0,
        "holdoutPeriods": holdout,
    }


def is_significant(p: float | None) -> bool:
    return p is not None and not np.isnan(p) and p < STATS_SIGNIFICANCE_LEVEL


def strength_label(r: float) -> str:
    a = abs(r)
    if a >= 0.7:
        return "strong"
    if a >= 0.4:
        return "moderate"
    if a >= 0.2:
        return "weak"
    return "negligible"


def ols_clustered(y: pd.Series, X: pd.DataFrame, groups: pd.Series) -> dict[str, Any]:
    """Pooled OLS with cluster-robust (by group, e.g. company) standard errors.

    For panels (company-years) where errors correlate within a company over
    time. CR1 small-sample correction: G/(G-1) * (N-1)/(N-K).
    """
    data = pd.concat([y.rename("__y"), X, groups.rename("__g")], axis=1).dropna()
    n, k = len(data), X.shape[1] + 1
    g_ids = data["__g"].unique()
    G = len(g_ids)
    if n < max(STATS_MIN_OBSERVATIONS, k + 3) or G < 3:
        raise ValueError(
            f"Need >= {max(STATS_MIN_OBSERVATIONS, k + 3)} observations across >= 3 groups (got n={n}, groups={G})"
        )
    yv = data["__y"].to_numpy(dtype=float)
    Xv = np.column_stack([np.ones(n), data[X.columns].to_numpy(dtype=float)])
    xtx_inv = np.linalg.pinv(Xv.T @ Xv)
    beta = xtx_inv @ Xv.T @ yv
    resid = yv - Xv @ beta
    meat = np.zeros((k, k))
    gv = data["__g"].to_numpy()
    for g in g_ids:
        mask = gv == g
        sg = Xv[mask].T @ resid[mask]
        meat += np.outer(sg, sg)
    cov = xtx_inv @ meat @ xtx_inv * (G / (G - 1)) * ((n - 1) / (n - k))
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, beta / se, np.nan)
    pvals = 2 * stats.t.sf(np.abs(t), df=G - 1)
    ss_res = float(resid @ resid)
    ss_tot = float(((yv - yv.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "coefficients": pd.DataFrame(
            {"coef": beta, "seCluster": se, "t": t, "p": pvals}, index=["const", *X.columns]
        ),
        "r2": r2,
        "n": n,
        "groups": G,
    }
