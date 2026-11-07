# src/data/series_loader.py
#
# One entry point that turns a catalog key or an ad-hoc spec into a dated
# pandas Series, regardless of source. This is what makes the statistical
# engine source-agnostic: FRED, World Bank, FMP prices and FMP sector/industry
# P/E all come out as the same `SeriesData` shape.
#
# Spec syntax (for anything not in config.MACRO_SERIES_CATALOG):
#   "fred:DGS10"                      any FRED series ID
#   "worldbank:CHN:FR.INR.LEND"       any World Bank country/indicator
#   "price:TSLA"                      FMP daily close
#   "sectorpe:Technology"             FMP historical sector P/E (NASDAQ)
#   "industrype:Auto - Manufacturers" FMP historical industry P/E (NASDAQ)

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from src.config import MACRO_SERIES_CATALOG, RELVAL_HISTORY_YEARS
from src.data import store
from src.data.fmp_client import FMPClient, FMPError
from src.data.fred_client import FREDClient, FREDError
from src.data.worldbank_client import WorldBankClient, WorldBankError
from src.utils.logging import get_logger

logger = get_logger("series_loader")


class SeriesLoadError(Exception):
    """A requested series could not be loaded (bad spec, missing data, API failure)."""


@dataclass
class SeriesData:
    key: str
    label: str
    series: pd.Series  # DatetimeIndex, float values, sorted ascending, no NaN
    unit: str  # "pct" (percentage points) | "index" | "level" | "ratio"
    freq: str  # native frequency: D / M / Q / A
    source: str  # human-readable provenance for methodology/citations
    notes: list[str] = field(default_factory=list)


@dataclass
class _Clients:
    fred: FREDClient | None = None
    fmp: FMPClient | None = None
    wb: WorldBankClient | None = None

    def get_fred(self) -> FREDClient:
        self.fred = self.fred or FREDClient()
        return self.fred

    def get_fmp(self) -> FMPClient:
        self.fmp = self.fmp or FMPClient()
        return self.fmp

    def get_wb(self) -> WorldBankClient:
        self.wb = self.wb or WorldBankClient()
        return self.wb


def _to_series(rows: list[dict], value_key: str = "value") -> pd.Series:
    if not rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame(rows)
    s = pd.Series(pd.to_numeric(df[value_key], errors="coerce").values, index=pd.to_datetime(df["date"]))
    return s.dropna().sort_index().groupby(level=0).last()


def _resolve(key_or_spec: str) -> dict[str, str]:
    if key_or_spec in MACRO_SERIES_CATALOG:
        return {"key": key_or_spec, **MACRO_SERIES_CATALOG[key_or_spec]}
    source, sep, ident = key_or_spec.partition(":")
    if not sep or not ident.strip():
        raise SeriesLoadError(
            f"Unknown series {key_or_spec!r}: use a catalog key or 'source:id' "
            "(fred:, worldbank:, price:, sectorpe:, industrype:)"
        )
    source = source.strip().lower()
    ident = ident.strip()
    defaults = {
        "fred": ("level", "M", f"FRED {ident}"),
        "worldbank": ("level", "A", f"World Bank {ident}"),
        "price": ("level", "D", f"{ident.upper()} share price"),
        "sectorpe": ("ratio", "D", f"{ident} sector P/E"),
        "industrype": ("ratio", "D", f"{ident} industry P/E"),
    }
    if source not in defaults:
        raise SeriesLoadError(f"Unknown series source {source!r} in {key_or_spec!r}")
    unit, freq, label = defaults[source]
    return {"key": key_or_spec, "source": source, "id": ident, "label": label, "unit": unit, "freq": freq}


def load_series(key_or_spec: str, start: str | None = None, clients: _Clients | None = None) -> SeriesData:
    """Load one series as `SeriesData`. Raises SeriesLoadError on any failure."""
    spec = _resolve(key_or_spec.strip())
    clients = clients or _Clients()
    source, ident = spec["source"], spec["id"]
    notes: list[str] = []
    try:
        if source == "fred":
            client = clients.get_fred()
            rows = client.get_history(ident, start=start)
            provenance = f"FRED series {ident}" + (" (public CSV)" if client.keyless else "")
        elif source == "worldbank":
            country, _, indicator = ident.partition(":")
            rows = [
                r
                for r in clients.get_wb().get_indicator(country, indicator)
                if not start or r["date"] >= start
            ]
            provenance = f"World Bank WDI {indicator} ({country})"
        elif source == "price":
            fmp = clients.get_fmp()
            rows = fmp.get_historical_prices(ident, start=start)
            rows = [{"date": r["date"], "value": r.get("price")} for r in rows]
            provenance = f"FMP daily close, {ident.upper()}"
        elif source in ("sectorpe", "industrype"):
            fmp = clients.get_fmp()
            begin = start or f"{date.today().year - RELVAL_HISTORY_YEARS}-01-01"
            fetch = fmp.get_historical_sector_pe if source == "sectorpe" else fmp.get_historical_industry_pe
            rows = [
                {"date": r["date"], "value": r.get("pe")}
                for r in fetch(ident, begin, date.today().isoformat())
            ]
            provenance = (
                f"FMP historical {'sector' if source == 'sectorpe' else 'industry'} P/E, {ident} (NASDAQ)"
            )
            notes.append(
                "FMP aggregate P/E methodology is not published; cap-weighted aggregates can be skewed by mega-caps."
            )
        else:  # pragma: no cover — _resolve guards this
            raise SeriesLoadError(source)
    except (FREDError, WorldBankError, FMPError, ValueError) as exc:
        # Offline resilience: fall back to the last copy in the Parquet store.
        stored = store.load_series(source, ident)
        if stored is None or stored.empty:
            raise SeriesLoadError(f"{spec['label']}: {exc}") from exc
        if start:
            stored = stored.loc[pd.Timestamp(start) :]
            if stored.empty:
                raise SeriesLoadError(
                    f"{spec['label']}: live source failed ({exc}) and the local store has no data after {start}"
                ) from exc
        logger.warning("%s: live source failed (%s) — serving local store copy", spec["label"], exc)
        notes.append(
            f"Live source unavailable ({exc}); served from the local Parquet store, last observation "
            f"{stored.index[-1]:%Y-%m-%d}."
        )
        return SeriesData(
            key=spec["key"],
            label=spec["label"],
            series=stored,
            unit=spec["unit"],
            freq=spec["freq"],
            source=f"local store copy of {source}:{ident}",
            notes=notes,
        )

    series = _to_series(rows)
    if series.empty:
        raise SeriesLoadError(f"{spec['label']}: no observations returned")
    try:
        store.upsert_series(source, ident, series, label=spec["label"])  # write-through
    except OSError as exc:
        logger.warning("Could not persist %s to store: %s", spec["label"], exc)
    return SeriesData(
        key=spec["key"],
        label=spec["label"],
        series=series.astype(float),
        unit=spec["unit"],
        freq=spec["freq"],
        source=provenance,
        notes=notes,
    )


def load_many(keys: list[str], start: str | None = None) -> tuple[list[SeriesData], dict[str, str]]:
    """Load several series; one failure never kills the batch.
    Returns (loaded series, {key: error message} for failures).
    """
    if not keys:
        raise ValueError("keys must be a non-empty list")
    clients = _Clients()
    loaded: list[SeriesData] = []
    errors: dict[str, str] = {}
    for key in dict.fromkeys(keys):  # de-duplicate, keep order
        try:
            loaded.append(load_series(key, start=start, clients=clients))
        except SeriesLoadError as exc:
            logger.warning("Series %s skipped: %s", key, exc)
            errors[key] = str(exc)
    return loaded, errors
