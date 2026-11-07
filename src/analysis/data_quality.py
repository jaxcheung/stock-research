# src/analysis/data_quality.py
#
# Deterministic checks on statement data before it feeds an analysis. First
# check: stub rows — FMP sometimes publishes a latest-period statement with a
# handful of fields filled and the rest zero (RJF FY2026 balance sheet: 3
# non-zero fields, total assets $155m vs ~$85bn a year earlier, equity 0). Read
# as real, that produced "negative shareholders' equity ($0.00)" and a 10/100
# health score. A stub is never silently "fixed": the caller falls back to the
# prior period and must surface the returned note.

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from src.config import (
    DQ_IDENTITY_TOLERANCE,
    DQ_MAX_GAP_DAYS,
    DQ_REVENUE_JUMP,
    DQ_SHARE_CHANGE,
    DQ_TIEOUT_WARN,
    DQ_UNMAPPED_MATERIALITY,
    STUB_STATEMENT_ASSET_RATIO,
)


def is_stub_balance(row: dict[str, Any], prior: dict[str, Any] | None) -> str | None:
    """Reason string if `row` looks like a stub, else None."""
    assets = row.get("totalAssets") or 0.0
    equity = row.get("totalStockholdersEquity") or 0.0
    liabilities = row.get("totalLiabilities") or 0.0
    if assets > 0 and equity == 0 and liabilities == 0:
        return "equity and liabilities are both zero"
    prior_assets = (prior or {}).get("totalAssets") or 0.0
    if prior_assets > 0 and assets < STUB_STATEMENT_ASSET_RATIO * prior_assets:
        return f"total assets {assets:,.0f} are under {STUB_STATEMENT_ASSET_RATIO:.0%} of the prior period's {prior_assets:,.0f}"
    return None


def latest_complete_balance(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], str | None]:
    """(balance sheet to use, data-quality note or None). `rows` is newest-first."""
    if not rows:
        return {}, None
    latest, prior = rows[0], rows[1] if len(rows) > 1 else None
    why = is_stub_balance(latest, prior)
    if why and prior:
        return prior, (
            f"Latest balance sheet ({latest.get('date')}) looks like an incomplete stub ({why}); "
            f"using {prior.get('date')} instead."
        )
    if why:
        return (
            latest,
            f"Latest balance sheet ({latest.get('date')}) looks incomplete ({why}); no prior period to fall back to.",
        )
    return latest, None


# =========================================================================== statement checks


@dataclass
class Issue:
    severity: str  # "warn" (affects numbers you'd rely on) | "info" (context worth knowing)
    check: str  # short check name
    period: str | None
    message: str


@dataclass
class DataQualityReport:
    issues: list[Issue] = field(default_factory=list)
    periods_checked: int = 0

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "warn"]

    @property
    def infos(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "info"]

    def summary(self) -> str:
        if not self.issues:
            return f"No data-quality issues in {self.periods_checked} annual periods checked."
        return (
            f"{len(self.warnings)} warning(s), {len(self.infos)} note(s) across "
            f"{self.periods_checked} annual periods checked."
        )

    def lines(self) -> list[str]:
        return [
            f"{'⚠️' if i.severity == 'warn' else 'ℹ️'} [{i.check}{' · ' + i.period if i.period else ''}] {i.message}"
            for i in self.issues
        ]


def _num(row: dict[str, Any], key: str) -> float | None:
    v = row.get(key)
    return float(v) if isinstance(v, int | float) else None


def _period(row: dict[str, Any]) -> str:
    return str(row.get("fiscalYear") or str(row.get("date", ""))[:4] or "?")


def _by_date(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(r.get("date")): r for r in rows if r.get("date")}


def _check_identities(
    rep: DataQualityReport,
    balance: list[dict[str, Any]],
    cash: list[dict[str, Any]],
    income: list[dict[str, Any]],
) -> None:
    tol = DQ_IDENTITY_TOLERANCE
    for i, b in enumerate(balance):
        prior = balance[i + 1] if i + 1 < len(balance) else None
        if is_stub_balance(b, prior):
            continue
        assets = _num(b, "totalAssets")
        liab = _num(b, "totalLiabilities")
        equity = _num(b, "totalEquity")
        if equity is None:
            se, mi = _num(b, "totalStockholdersEquity"), _num(b, "minorityInterest") or 0.0
            equity = se + mi if se is not None else None
        if (
            assets
            and liab is not None
            and equity is not None
            and abs(assets - liab - equity) > tol * abs(assets)
        ):
            rep.issues.append(
                Issue(
                    "warn",
                    "balance identity",
                    _period(b),
                    f"assets {assets:,.0f} ≠ liabilities + equity {liab + equity:,.0f} "
                    f"({(assets - liab - equity) / assets:+.1%} of assets) — a field is missing or mis-mapped",
                )
            )
    for c in cash:
        cfo, capex, fcf = _num(c, "operatingCashFlow"), _num(c, "capitalExpenditure"), _num(c, "freeCashFlow")
        if cfo and capex is not None and fcf is not None:
            # FMP reports capex as a negative cash flow; FCF = CFO + capex.
            expected = cfo + capex if capex <= 0 else cfo - capex
            if abs(fcf - expected) > tol * abs(cfo):
                rep.issues.append(
                    Issue(
                        "warn",
                        "free cash flow",
                        _period(c),
                        f"FCF {fcf:,.0f} ≠ operating cash flow − capex {expected:,.0f} — DCF inputs may be inconsistent",
                    )
                )
    inc_by_date = _by_date(income)
    for c in cash:
        if (_num(c, "operatingCashFlow") or 0.0) == 0 and (_num(c, "capitalExpenditure") or 0.0) == 0:
            continue  # stub cash-flow row — flagged by the stub check, not a tie-out problem
        i_row = inc_by_date.get(str(c.get("date")))
        ni_i, ni_c = (_num(i_row, "netIncome") if i_row else None), _num(c, "netIncome")
        if ni_i and ni_c is not None and abs(ni_i - ni_c) > tol * abs(ni_i):
            gap = abs(ni_i - ni_c) / abs(ni_i)
            large = gap > DQ_TIEOUT_WARN
            rep.issues.append(
                Issue(
                    "warn" if large else "info",
                    "net income tie-out",
                    _period(c),
                    f"income statement {ni_i:,.0f} vs cash-flow statement {ni_c:,.0f} ({gap:.0%} apart) — "
                    + (
                        "too large for minority interests alone: discontinued operations, a restatement or a "
                        "mis-mapped field; check the filing before relying on earnings-based ratios"
                        if large
                        else "the cash-flow statement often starts from consolidated income incl. minority interests"
                    ),
                )
            )


def _check_stubs(
    rep: DataQualityReport, name: str, rows: list[dict[str, Any]], core: tuple[str, ...]
) -> None:
    if len(rows) < 2:
        return
    latest, prior = rows[0], rows[1]
    zero_now = all((_num(latest, k) or 0.0) == 0 for k in core)
    material_before = any((_num(prior, k) or 0.0) != 0 for k in core)
    if zero_now and material_before:
        rep.issues.append(
            Issue(
                "warn",
                "stub row",
                _period(latest),
                f"latest {name} has {', '.join(core)} all zero while the prior year did not — an incomplete "
                "filing; analyses fall back to the prior period where they can",
            )
        )


def _check_completeness(
    rep: DataQualityReport, name: str, rows: list[dict[str, Any]], expected: int | None
) -> None:
    years = [r.get("fiscalYear") for r in rows if r.get("fiscalYear") is not None]
    dupes = sorted({str(y) for y in years if years.count(y) > 1})
    if dupes:
        rep.issues.append(
            Issue("warn", "duplicate period", None, f"{name}: fiscal year(s) {', '.join(dupes)} appear twice")
        )
    dates = sorted(date.fromisoformat(str(r["date"])[:10]) for r in rows if r.get("date"))
    for a, b in zip(dates, dates[1:], strict=False):
        if (b - a).days > DQ_MAX_GAP_DAYS:
            rep.issues.append(
                Issue(
                    "warn",
                    "missing period",
                    str(b.year),
                    f"{name}: {(b - a).days} days between {a} and {b} — a year is missing",
                )
            )
    if expected and len(rows) < expected:
        rep.issues.append(
            Issue(
                "info", "short history", None, f"{name}: {len(rows)} of {expected} requested years available"
            )
        )


def _check_plausibility(
    rep: DataQualityReport, income: list[dict[str, Any]], balance: list[dict[str, Any]]
) -> None:
    for cur, prev in zip(income, income[1:], strict=False):
        r1, r0 = _num(cur, "revenue"), _num(prev, "revenue")
        if r1 and r0 and r1 > 0 and r0 > 0 and (r1 / r0 > DQ_REVENUE_JUMP or r1 / r0 < 1 / DQ_REVENUE_JUMP):
            rep.issues.append(
                Issue(
                    "info",
                    "revenue jump",
                    _period(cur),
                    f"revenue {r0:,.0f} → {r1:,.0f} ({r1 / r0:.1f}×): M&A, restatement or units?",
                )
            )
        s1, s0 = _num(cur, "weightedAverageShsOutDil"), _num(prev, "weightedAverageShsOutDil")
        if s1 and s0 and abs(s1 / s0 - 1) > DQ_SHARE_CHANGE:
            rep.issues.append(
                Issue(
                    "info",
                    "share count",
                    _period(cur),
                    f"diluted shares {s0:,.0f} → {s1:,.0f} ({s1 / s0 - 1:+.0%}): split, merger or large buyback? "
                    "Per-share history is not split-adjusted here",
                )
            )
    for cur, prev in zip(balance, balance[1:], strict=False):
        e1, e0 = _num(cur, "totalStockholdersEquity"), _num(prev, "totalStockholdersEquity")
        if e1 is not None and e0 is not None and e0 > 0 > e1 and not is_stub_balance(cur, prev):
            rep.issues.append(
                Issue(
                    "info",
                    "equity sign",
                    _period(cur),
                    "shareholders' equity turned negative — D/E and ROE are not meaningful",
                )
            )


def _check_unmapped(
    rep: DataQualityReport, income: list[dict[str, Any]], balance: list[dict[str, Any]]
) -> None:
    m = DQ_UNMAPPED_MATERIALITY
    if len(income) >= 2:
        latest, prior = income[0], income[1]
        rev = _num(prior, "revenue") or 0.0
        for k in ("interestIncome", "interestExpense"):
            now, before = _num(latest, k), _num(prior, k)
            if now == 0 and before and rev and abs(before) > m * rev:
                rep.issues.append(
                    Issue(
                        "warn",
                        "unmapped field",
                        _period(latest),
                        f"{k} is 0 but was {before:,.0f} last year — likely not mapped by the data provider",
                    )
                )
    if len(balance) >= 2 and not is_stub_balance(balance[0], balance[1]):
        latest, prior = balance[0], balance[1]
        assets = _num(prior, "totalAssets") or 0.0
        now, before = _num(latest, "totalDebt"), _num(prior, "totalDebt")
        if now == 0 and before and assets and before > m * assets:
            rep.issues.append(
                Issue(
                    "warn",
                    "unmapped field",
                    _period(latest),
                    f"totalDebt is 0 but was {before:,.0f} last year — likely not mapped",
                )
            )


def _check_currency(rep: DataQualityReport, *statements: list[dict[str, Any]]) -> None:
    ccys = {str(r.get("reportedCurrency")) for rows in statements for r in rows if r.get("reportedCurrency")}
    if len(ccys) > 1:
        rep.issues.append(
            Issue(
                "warn",
                "reporting currency",
                None,
                f"statements are reported in more than one currency ({', '.join(sorted(ccys))}) — cross-year and cross-statement ratios mix currencies",
            )
        )


def check_statements(
    income: list[dict[str, Any]],
    balance: list[dict[str, Any]],
    cashflow: list[dict[str, Any]],
    expected_years: int | None = None,
) -> DataQualityReport:
    """Deterministic consistency checks on annual statements (each list newest-first).
    Returns issues to SURFACE; nothing here changes the data."""
    rep = DataQualityReport(periods_checked=len(income))
    _check_currency(rep, income, balance, cashflow)
    _check_stubs(rep, "balance sheet", balance, ("totalLiabilities", "totalStockholdersEquity"))
    _check_stubs(rep, "cash-flow statement", cashflow, ("operatingCashFlow", "capitalExpenditure"))
    _check_stubs(rep, "income statement", income, ("revenue", "netIncome"))
    for name, rows in (
        ("income statement", income),
        ("balance sheet", balance),
        ("cash-flow statement", cashflow),
    ):
        _check_completeness(rep, name, rows, expected_years)
    _check_identities(rep, balance, cashflow, income)
    _check_plausibility(rep, income, balance)
    _check_unmapped(rep, income, balance)
    return rep
