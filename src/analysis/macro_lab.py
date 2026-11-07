# src/analysis/macro_lab.py
#
# Multi-variable research orchestrator: load N series from any source, align
# them, and run the full statistical battery (descriptives, stationarity,
# correlation on levels AND stationary transforms, lead/lag, HAC regression,
# cycle decomposition, backtested forecast). Returns a structured report —
# data, findings in plain language, methodology, sources, caveats — that
# the dashboard and HTML exporter both render. No plotting here.

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.analysis import timeseries as ts
from src.config import STATS_MAX_LAG, STATS_MIN_OBSERVATIONS, STATS_SIGNIFICANCE_LEVEL
from src.data.series_loader import SeriesData, load_many
from src.utils.logging import get_logger

logger = get_logger("macro_lab")


@dataclass
class Finding:
    text: str
    tone: str = "neutral"  # neutral | positive | caution


@dataclass
class MacroLabReport:
    series: list[SeriesData]
    freq: str
    transform: str
    levels: pd.DataFrame
    transformed: pd.DataFrame
    stats: pd.DataFrame
    corr_levels: dict[str, pd.DataFrame]
    corr_transformed: dict[str, pd.DataFrame]
    recommended_basis: str  # "levels" | "transformed"
    target: str | None
    # ADF result per variable on the basis actually used for inference (None = untestable)
    basis_stationary: dict[str, bool | None] = field(default_factory=dict)
    lead_lag: dict[str, pd.DataFrame] = field(default_factory=dict)
    regression: dict[str, Any] | None = None
    cycles: dict[str, dict[str, Any]] = field(default_factory=dict)
    forecast: dict[str, Any] | None = None
    findings: list[Finding] = field(default_factory=list)
    methodology: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    load_errors: dict[str, str] = field(default_factory=dict)

    @property
    def labels(self) -> dict[str, str]:
        return {s.label: s.label for s in self.series}

    @property
    def sources(self) -> list[str]:
        return [f"{s.label} — {s.source}" for s in self.series]


def _fmt_r(r: float, p: float) -> str:
    return f"r = {r:+.2f} (p {'< 0.001' if p < 0.001 else f'= {p:.3f}'})"


def run_macro_lab(
    keys: list[str],
    freq: str = "auto",
    transform: str = "yoy",
    start: str | None = None,
    end: str | None = None,
    target: str | None = None,
    horizon: int = 12,
    method: str = "pearson",
    max_lag: int = STATS_MAX_LAG,
) -> MacroLabReport:
    """Run the full multi-variable analysis.

    keys: catalog keys or specs (see series_loader). target: catalog key of
    the dependent variable for lead/lag, regression and forecast (defaults
    to the first series). Raises ValueError if fewer than one series loads
    or the series don't overlap.
    """
    if not keys:
        raise ValueError("Select at least one series")
    if transform not in ts.TRANSFORMS:
        raise ValueError(f"transform must be one of {list(ts.TRANSFORMS)}")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")

    loaded, errors = load_many(keys, start=start)
    if not loaded:
        raise ValueError("No series could be loaded: " + "; ".join(errors.values()))

    by_label = {s.label: s for s in loaded}
    levels, used_freq, align_notes = ts.align(
        {s.label: s.series for s in loaded},
        {s.label: s.freq for s in loaded},
        freq=freq,
        how="mean",
        start=start,
        end=end,
    )
    transformed = ts.transform(levels, transform, used_freq)
    stats_df = ts.descriptive_stats(levels, used_freq)

    corr_levels = ts.correlation_matrix(levels, method) if levels.shape[1] > 1 else {}
    corr_tr = ts.correlation_matrix(transformed, method) if transformed.shape[1] > 1 else {}
    # Levels are a valid inference basis only when ADF positively establishes stationarity
    # for every series; "untestable" (too few observations) is not a pass.
    non_stationary = [v for v, row in stats_df.iterrows() if row["stationary"] is not True]
    recommended = "transformed" if non_stationary and transform != "level" else "levels"
    basis_for_adf = transformed if recommended == "transformed" else levels
    basis_stationary = {str(c): ts.adf_test(basis_for_adf[c])["stationary"] for c in basis_for_adf.columns}

    target_label = next((s.label for s in loaded if s.key == target), loaded[0].label)
    report = MacroLabReport(
        series=loaded,
        freq=used_freq,
        transform=transform,
        levels=levels,
        transformed=transformed,
        stats=stats_df,
        corr_levels=corr_levels,
        corr_transformed=corr_tr,
        recommended_basis=recommended,
        basis_stationary=basis_stationary,
        target=target_label,
        load_errors=errors,
    )
    report.caveats.extend(align_notes)
    common_end = levels.index[-1]
    limiting = [s.label for s in loaded if (s.series.index[-1] - common_end).days < 45]
    latest = max(s.series.index[-1] for s in loaded)
    if (latest - common_end).days > 180 and limiting:
        report.caveats.append(
            f"Common sample ends {common_end:%b %Y} because {', '.join(limiting)} "
            f"{'is' if len(limiting) == 1 else 'are'} not published beyond that; other series run to "
            f"{latest:%b %Y}. Drop the limiting series to analyse the most recent data."
        )
    for s in loaded:
        report.caveats.extend(f"{s.label}: {n}" for n in s.notes)
    if errors:
        report.caveats.append("Not loaded: " + "; ".join(f"{k} ({v})" for k, v in errors.items()))

    _add_descriptive_findings(report)
    drivers = [c for c in levels.columns if c != target_label]
    basis_df = transformed if recommended == "transformed" else levels

    # Lead/lag of each driver vs the target (on the stationary basis).
    for d in drivers:
        try:
            report.lead_lag[d] = ts.lead_lag(basis_df, d, target_label, max_lag=max_lag)
        except ValueError as exc:
            logger.warning("Lead/lag %s vs %s skipped: %s", d, target_label, exc)

    _add_correlation_findings(report, basis_df)

    # Multivariate regression: target ~ drivers (stationary basis, HAC SEs).
    reg_vars_ns = [v for v in [target_label, *drivers] if report.basis_stationary.get(v) is not True]
    if drivers and reg_vars_ns:
        report.caveats.append(
            "Regression not run: stationarity of " + ", ".join(reg_vars_ns) + " is not established on the "
            f"'{ts.TRANSFORMS[report.transform if report.recommended_basis == 'transformed' else 'level']}' "
            "basis (ADF fails, or too few observations to test) — a regression of possibly non-stationary "
            "series can be spurious. Use a differencing transform and/or a longer sample."
        )
    elif drivers and len(basis_df) >= STATS_MIN_OBSERVATIONS:
        try:
            report.regression = ts.ols_hac(basis_df[target_label], basis_df[drivers])
            _add_regression_findings(report, drivers)
        except ValueError as exc:
            report.caveats.append(f"Regression not run: {exc}")

    # Cycle analysis for every series; forecast for the target.
    for col in levels.columns:
        try:
            report.cycles[col] = ts.cycle_analysis(
                levels[col], used_freq, log_scale=by_label[col].unit in ("level", "index")
            )
        except ValueError as exc:
            report.caveats.append(f"Cycle analysis skipped for {col}: {exc}")
    _add_cycle_findings(report)

    if used_freq != "D":
        try:
            report.forecast = ts.forecast(
                levels[target_label],
                used_freq,
                horizon,
                log_scale=by_label[target_label].unit in ("level", "index"),
            )
            _add_forecast_findings(report, by_label[target_label])
        except ValueError as exc:
            report.caveats.append(f"Forecast not produced: {exc}")

    report.methodology = _methodology(report, method, max_lag)
    return report


# --------------------------------------------------------------------------- findings


def _add_descriptive_findings(rep: MacroLabReport) -> None:
    for var, row in rep.stats.iterrows():
        pct = row["lastPercentile"]
        if pct >= 90 or pct <= 10:
            where = "high" if pct >= 90 else "low"
            rep.findings.append(
                Finding(
                    f"**{var}** is at a historically {where} level — the {pct:.0f}th percentile of its "
                    f"{row['start'][:4]}–{row['end'][:4]} range (z = {row['lastZScore']:+.1f}).",
                    "caution",
                )
            )
    ns = [str(v) for v, r in rep.stats.iterrows() if r["stationary"] is False]
    if ns and rep.levels.shape[1] > 1:
        still_ns = [v for v, ok in rep.basis_stationary.items() if ok is False]
        if rep.recommended_basis == "levels":
            tail = (
                "No transform was chosen, so no correlation, lead/lag or regression inference is "
                "reported for these series."
            )
        elif still_ns:
            tail = (
                f"The '{ts.TRANSFORMS[rep.transform]}' transform does not make "
                + ", ".join(still_ns)
                + " stationary either (ADF, 5%), so no inference is reported for them — choose a "
                "differencing transform (Δ, % change, YoY)."
            )
        else:
            tail = (
                f"Inference uses the '{ts.TRANSFORMS[rep.transform]}' transform instead, which passes "
                "the ADF test for every series."
            )
        rep.caveats.append(
            "Non-stationary in levels (ADF, 5%): "
            + ", ".join(ns)
            + ". Level correlations between trending series can be spurious. "
            + tail
        )


def _add_correlation_findings(rep: MacroLabReport, basis_df: pd.DataFrame) -> None:
    corr = rep.corr_transformed if rep.recommended_basis == "transformed" else rep.corr_levels
    if not corr:
        return
    r, p = corr["r"], corr["p"]
    cols = [str(c) for c in r.columns]
    r_np, p_np = r.to_numpy(dtype=float), p.to_numpy(dtype=float)
    pairs = []
    for i, a in enumerate(cols):
        for j in range(i + 1, len(cols)):
            b = cols[j]
            rv, pv = float(r_np[i, j]), float(p_np[i, j])
            if rep.basis_stationary.get(a) is not True or rep.basis_stationary.get(b) is not True:
                continue  # no inference unless stationarity is established
            if pd.notna(rv) and ts.is_significant(pv):
                pairs.append((abs(rv), a, b, rv, pv))
    for _, a, b, rv, pv in sorted(pairs, reverse=True)[:4]:
        direction = "move together" if rv > 0 else "move inversely"
        rep.findings.append(
            Finding(
                f"**{a}** and **{b}** {direction} — {ts.strength_label(rv)} {_fmt_r(rv, pv)}.", "positive"
            )
        )
    if not pairs and any(ok is not True for ok in rep.basis_stationary.values()):
        untestable = [v for v, ok in rep.basis_stationary.items() if ok is None]
        rep.findings.append(
            Finding(
                "No correlation inference reported for series whose stationarity is not established on the "
                "chosen basis"
                + (f" ({', '.join(untestable)}: too few observations for an ADF test)" if untestable else "")
                + ".",
                "caution",
            )
        )
    elif not pairs:
        rep.findings.append(
            Finding(
                f"No statistically significant pairwise correlation at the {STATS_SIGNIFICANCE_LEVEL:.0%} level."
            )
        )
    # Spurious-correlation guard: strong in levels, gone after transform.
    if rep.corr_levels and rep.corr_transformed:
        rl_np = rep.corr_levels["r"].to_numpy(dtype=float)
        rt_np = rep.corr_transformed["r"].to_numpy(dtype=float)
        for i, a in enumerate(cols):
            for j in range(i + 1, len(cols)):
                b = cols[j]
                rl, rt = float(rl_np[i, j]), float(rt_np[i, j])
                if pd.notna(rl) and pd.notna(rt) and abs(rl) >= 0.6 and abs(rt) < 0.2:
                    rep.caveats.append(
                        f"{a} vs {b}: r = {rl:+.2f} in levels but {rt:+.2f} after transform — the level "
                        "relationship is likely shared trend, not a genuine link."
                    )
    for driver, ll in rep.lead_lag.items():
        if (
            rep.basis_stationary.get(driver) is not True
            or rep.basis_stationary.get(str(rep.target)) is not True
        ):
            continue
        best = ts.best_lag(ll)
        if best and ts.is_significant(best["p"]) and best["lag"] != 0 and abs(best["r"]) >= 0.2:
            lead = best["lag"] > 0
            k = abs(best["lag"])
            unit = {"M": "month", "Q": "quarter", "A": "year", "W": "week", "D": "day"}[rep.freq]
            if k == max_lag_used(ll):
                rep.caveats.append(
                    f"{driver} vs {rep.target}: peak cross-correlation sits at the ±{k}-period search "
                    "boundary — the true lead/lag may be longer; widen the lag window to confirm."
                )
            rep.findings.append(
                Finding(
                    f"**{driver}** {'leads' if lead else 'lags'} **{rep.target}** by {k} {unit}{'s' if k > 1 else ''} "
                    f"(peak cross-correlation {_fmt_r(best['r'], best['p'])}).",
                    "positive",
                )
            )


def max_lag_used(ll: pd.DataFrame) -> int:
    return int(ll["lag"].abs().max()) if not ll.empty else 0


def _add_regression_findings(rep: MacroLabReport, drivers: list[str]) -> None:
    reg = rep.regression
    if not reg:
        return
    coefs = reg["coefficients"]
    sig = [d for d in drivers if ts.is_significant(coefs.loc[d, "p"])]
    rep.findings.append(
        Finding(
            f"Multivariate model explains {reg['r2']:.0%} of the variation in **{rep.target}** "
            f"(adj. R² {reg['adjR2']:.2f}, n = {reg['n']}); significant drivers: "
            + (", ".join(f"{d} (β = {coefs.loc[d, 'coef']:+.2f})" for d in sig) if sig else "none")
            + ".",
            "positive" if sig else "neutral",
        )
    )
    if reg["durbinWatson"] < 1.0:
        rep.caveats.append(
            f"Durbin-Watson = {reg['durbinWatson']:.2f}: strong residual autocorrelation. HAC standard errors "
            "correct the t-stats, but the model is likely missing dynamics (lags)."
        )


def _add_cycle_findings(rep: MacroLabReport) -> None:
    for var, c in rep.cycles.items():
        period = c["dominantPeriodYears"]
        text = f"**{var}** cycle phase: **{c['phase']}** ({c['phaseDescription']})"
        if period:
            text += f"; dominant cycle ≈ {period:.1f} years"
            if c["businessCycleBandShare"] is not None:
                text += f" ({c['businessCycleBandShare']:.0%} of cycle variance in the 1.5–10y band)"
        rep.findings.append(Finding(text + ".", "neutral"))


def _add_forecast_findings(rep: MacroLabReport, target: SeriesData) -> None:
    fc = rep.forecast
    if not fc:
        return
    end_value = float(fc["forecast"].iloc[-1])
    horizon = len(fc["forecast"])
    if fc["beatsNaive"]:
        rep.findings.append(
            Finding(
                f"Cycle model projects **{rep.target}** at {end_value:,.2f} in {horizon} periods "
                f"(95% band {float(fc['lower95'].iloc[-1]):,.2f}–{float(fc['upper95'].iloc[-1]):,.2f}). "
                f"Out-of-sample skill vs naive: {fc['skill']:+.0%}.",
                "positive",
            )
        )
    else:
        rep.findings.append(
            Finding(
                f"Cycle model does **not** beat a naive forecast for **{rep.target}** out of sample "
                f"(skill {fc['skill']:+.0%}) — treat the projection as illustrative only.",
                "caution",
            )
        )


def _methodology(rep: MacroLabReport, method: str, max_lag: int) -> list[str]:
    unit = {"M": "monthly", "Q": "quarterly", "A": "annual", "W": "weekly", "D": "daily"}[rep.freq]
    hp = rep.cycles[next(iter(rep.cycles))]["hpLambda"] if rep.cycles else None
    steps = [
        f"**Data alignment** — {len(rep.series)} series resampled to a common {unit} frequency (period mean) "
        f"and inner-joined: {len(rep.levels)} overlapping observations, "
        f"{rep.levels.index[0]:%b %Y} – {rep.levels.index[-1]:%b %Y}. Coarser series are never interpolated "
        "unless a finer frequency is forced.",
        "**Stationarity** — Augmented Dickey-Fuller test (constant, Schwert lag rule, MacKinnon 5% critical "
        "value −2.86). Inference on non-stationary levels is avoided because trending series correlate "
        "spuriously (Granger & Newbold, 1974).",
        f"**Transform** — {ts.TRANSFORMS[rep.transform]} applied for correlation, lead/lag and regression "
        f"(basis used: {rep.recommended_basis}).",
        f"**Correlation** — {method.title()} coefficients with two-sided p-values; significance at "
        f"{STATS_SIGNIFICANCE_LEVEL:.0%}; pairs with fewer than {STATS_MIN_OBSERVATIONS} observations are not reported.",
        f"**Lead/lag** — cross-correlation corr(driverₜ, targetₜ₊ₖ) for k ∈ [−{max_lag}, {max_lag}]; "
        "a peak at k > 0 means the driver leads.",
        "**Regression** — OLS of the target on all drivers with Newey-West HAC standard errors "
        "(bandwidth ⌊4(n/100)^{2/9}⌋), robust to autocorrelation and heteroskedasticity.",
    ]
    if hp is not None:
        steps.append(
            f"**Cycle** — Hodrick-Prescott trend/cycle split (λ = {hp:,.0f}, Ravn-Uhlig scaling). Levels and "
            "indices are decomposed in logs (gap = % from trend); rates and ratios in their own units (gap in pp). Dominant period from a Hann-windowed periodogram "
            "restricted to the 1.5–10 year business-cycle band. Phase per the OECD business-cycle clock "
            "(gap sign × gap direction). Turning points require prominence ≥ 0.5σ of the cycle."
        )
    if rep.forecast:
        steps.append(
            "**Forecast** — local linear trend + dominant-cycle harmonic, fitted by least squares on the most "
            f"recent ~2.5 cycles. Validated on a {rep.forecast['holdoutPeriods']}-period hold-out (model re-fit "
            "without the hold-out) against a naive random-walk benchmark; skill = 1 − RMSE(model)/RMSE(naive). "
            "95% band from residual σ widened by √h."
        )
    return steps
