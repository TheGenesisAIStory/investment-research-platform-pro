"""ECB Data Portal SDMX 2.1 REST client.

The client targets the official ECB Data Portal API:

* datasets: https://data.ecb.europa.eu/data/datasets
* REST entry point: https://data-api.ecb.europa.eu/service/
* API help: https://data.ecb.europa.eu/help/api/data
* content negotiation: https://data.ecb.europa.eu/help/api/content-negotiation

It intentionally keeps the SDMX flow/key visible in every output so that
academic and regulatory research can trace each feature back to the official
source and exact query.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd

from ..data_platform import resolve_data_platform_roots, utc_now


LOGGER = logging.getLogger(__name__)

ECB_API_BASE_URL = "https://data-api.ecb.europa.eu/service"

ECB_ACCEPT_HEADERS = {
    "csv": "text/csv",
    "csvdata": "text/csv",
    "pivot-csv": "application/vnd.ecb.data+csv;version=1.0.0",
    "json": "application/vnd.sdmx.data+json;version=1.0.0-wd",
    "sdmx-json": "application/vnd.sdmx.data+json;version=1.0.0-wd",
}

ECB_QUERY_FORMATS = {
    "csv": "csvdata",
    "csvdata": "csvdata",
    "pivot-csv": "csvdata",
    "json": "jsondata",
    "sdmx-json": "jsondata",
}

ECB_SERIES_PRESETS: dict[str, dict[str, Any]] = {
    "hicp_euro_area_yoy": {
        "flow_ref": "ICP",
        "key": "M.U2.N.000000.4.ANR",
        "label": "Euro area HICP overall index, annual rate of change",
        "frequency": "monthly",
        "category": "inflation",
    },
    "policy_rate_mro": {
        "flow_ref": "FM",
        "key": "D.U2.EUR.4F.KR.MRR_FR.LEV",
        "label": "ECB main refinancing operations fixed rate",
        "frequency": "daily",
        "category": "policy_rates",
    },
    "policy_rate_deposit": {
        "flow_ref": "FM",
        "key": "D.U2.EUR.4F.KR.DFR.LEV",
        "label": "ECB deposit facility rate",
        "frequency": "daily",
        "category": "policy_rates",
    },
    "policy_rate_marginal_lending": {
        "flow_ref": "FM",
        "key": "D.U2.EUR.4F.KR.MLFR.LEV",
        "label": "ECB marginal lending facility rate",
        "frequency": "daily",
        "category": "policy_rates",
    },
    "m2_notional_stock_index": {
        "flow_ref": "BSI",
        "key": "M.U2.Y.V.M20.X.I.U2.2300.Z01.E",
        "label": "Euro area M2, MFI BSI index of notional stocks",
        "frequency": "monthly",
        "category": "mfi_balance_sheet",
    },
    "m3_notional_stock_index": {
        "flow_ref": "BSI",
        "key": "M.U2.Y.V.M30.X.I.U2.2300.Z01.E",
        "label": "Euro area M3, MFI BSI index of notional stocks",
        "frequency": "monthly",
        "category": "mfi_balance_sheet",
    },
}


@dataclass(frozen=True)
class EcbSeriesSpec:
    """Configured ECB SDMX series reference."""

    name: str
    flow_ref: str
    key: str
    label: str
    frequency: str
    category: str

    @property
    def endpoint(self) -> str:
        return f"{ECB_API_BASE_URL}/data/{self.flow_ref}/{self.key}"


def _normalise_format(format_name: str) -> str:
    return str(format_name or "csv").strip().lower().replace("_", "-")


def _safe_read_csv(text: str) -> pd.DataFrame:
    for kwargs in ({"sep": ","}, {"sep": ";"}, {"sep": None, "engine": "python"}):
        try:
            df = pd.read_csv(StringIO(text), **kwargs)
            if len(df.columns) > 1:
                return df
        except Exception:
            continue
    return pd.read_csv(StringIO(text))


def normalize_ecb_frame(df: pd.DataFrame, flow_ref: str, key: str, query_url: str, dataset_name: str | None = None) -> pd.DataFrame:
    """Add standard date/value/provenance columns to an ECB CSV/JSON frame."""
    if df.empty:
        out = df.copy()
    else:
        out = df.copy()
    time_col = next((c for c in out.columns if str(c).upper() in {"TIME_PERIOD", "TIME"}), None)
    value_col = next((c for c in out.columns if str(c).upper() in {"OBS_VALUE", "VALUE"}), None)
    key_col = next((c for c in out.columns if str(c).upper() == "KEY"), None)
    if time_col and "date" not in out.columns:
        out["date"] = pd.to_datetime(out[time_col], errors="coerce")
    if value_col and "value" not in out.columns:
        out["value"] = pd.to_numeric(out[value_col], errors="coerce")
    out["flow_ref"] = flow_ref
    out["series_key"] = out[key_col].astype(str) if key_col else key
    out["dataset"] = dataset_name or f"{flow_ref}.{key}"
    out["source"] = "ecb_data_portal_sdmx"
    out["source_url"] = query_url
    out["retrieved_at"] = utc_now()
    return out


def parse_sdmx_json(payload: dict[str, Any], flow_ref: str, key: str, query_url: str) -> pd.DataFrame:
    """Parse the common SDMX-JSON observation layout into a tidy frame.

    The ECB also supports CSV, which is preferred for broad research use. This
    JSON parser covers the standard compact SDMX-JSON shape and preserves raw
    index labels where the structure cannot be fully decoded.
    """
    rows: list[dict[str, Any]] = []
    structure = payload.get("structure", {})
    dimensions = structure.get("dimensions", {})
    series_dims = dimensions.get("series", [])
    obs_dims = dimensions.get("observation", [])
    obs_values = obs_dims[0].get("values", []) if obs_dims else []
    obs_labels = [str(v.get("id", i)) for i, v in enumerate(obs_values)]
    series_map = payload.get("dataSets", [{}])[0].get("series", {})
    for series_key, series_payload in series_map.items():
        parts = str(series_key).split(":")
        decoded: dict[str, Any] = {}
        for idx, part in enumerate(parts):
            if idx >= len(series_dims):
                continue
            dim = series_dims[idx]
            values = dim.get("values", [])
            try:
                decoded[dim.get("id", f"dim_{idx}")] = values[int(part)].get("id")
            except Exception:
                decoded[dim.get("id", f"dim_{idx}")] = part
        for obs_key, values in series_payload.get("observations", {}).items():
            obs_idx = int(str(obs_key).split(":")[0])
            rows.append(
                {
                    **decoded,
                    "TIME_PERIOD": obs_labels[obs_idx] if obs_idx < len(obs_labels) else obs_key,
                    "OBS_VALUE": values[0] if values else None,
                    "KEY": f"{flow_ref}.{key}",
                }
            )
    return normalize_ecb_frame(pd.DataFrame(rows), flow_ref, key, query_url)


class EcbClient:
    """Small, reusable ECB SDMX 2.1 client for macro and monetary datasets."""

    def __init__(
        self,
        financial_db_root: Path | str | None = None,
        output_root: Path | str | None = None,
        base_url: str = ECB_API_BASE_URL,
        timeout: int = 45,
        session: Any | None = None,
    ):
        roots = resolve_data_platform_roots(financial_db_root=financial_db_root, repo_output_root=output_root)
        self.financial_db_root = roots.financial_db
        self.output_root = roots.repo_output
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session
        self.base_dir = self.financial_db_root / "OfficialMacro" / "ECB"
        self.raw_dir = self.base_dir / "raw"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def specs(self) -> list[EcbSeriesSpec]:
        return [EcbSeriesSpec(name, **meta) for name, meta in ECB_SERIES_PRESETS.items()]

    def build_url(self, flow_ref: str, key: str) -> str:
        return f"{self.base_url}/data/{flow_ref}/{key}"

    def get_series(
        self,
        flow_ref: str,
        key: str,
        start: str | None = None,
        end: str | None = None,
        format: str = "csv",
        **params: Any,
    ) -> pd.DataFrame:
        """Retrieve a series from the ECB SDMX 2.1 REST API.

        Parameters map directly to the ECB API: ``start`` and ``end`` become
        ``startPeriod`` and ``endPeriod``; keyword arguments can include
        ``lastNObservations``, ``firstNObservations``, ``updatedAfter``,
        ``detail`` and ``includeHistory``. ``format`` supports ``csv`` and
        ``sdmx-json`` through both query parameter and Accept header.
        """
        import requests

        format_key = _normalise_format(format)
        query: dict[str, Any] = {k: v for k, v in params.items() if v is not None}
        if start is not None:
            query["startPeriod"] = start
        if end is not None:
            query["endPeriod"] = end
        if "format" not in query:
            query["format"] = ECB_QUERY_FORMATS.get(format_key, "csvdata")
        headers = {
            "Accept": ECB_ACCEPT_HEADERS.get(format_key, "text/csv"),
            "Accept-Encoding": "gzip, deflate",
            "User-Agent": "ResearchPlatform/1.0 EcbClient",
        }
        session = self.session or requests
        url = self.build_url(flow_ref, key)
        response = session.get(url, params=query, headers=headers, timeout=self.timeout)
        if response.status_code == 304:
            return pd.DataFrame(columns=["date", "value", "flow_ref", "series_key", "source_url"])
        response.raise_for_status()
        query_url = getattr(response, "url", url)
        content_type = str(response.headers.get("Content-Type", "")).lower()
        text = response.text
        if "json" in content_type or format_key in {"json", "sdmx-json"}:
            payload = response.json()
            return parse_sdmx_json(payload, flow_ref, key, query_url)
        df = _safe_read_csv(text)
        return normalize_ecb_frame(df, flow_ref, key, query_url)

    def get_preset(self, name: str, start: str | None = None, end: str | None = None, format: str = "csv", **params: Any) -> pd.DataFrame:
        spec_map = {spec.name: spec for spec in self.specs()}
        if name not in spec_map:
            raise KeyError(f"Unknown ECB preset: {name}")
        spec = spec_map[name]
        df = self.get_series(spec.flow_ref, spec.key, start=start, end=end, format=format, **params)
        df["dataset"] = name
        df["description"] = spec.label
        df["frequency"] = spec.frequency
        df["category"] = spec.category
        return df

    def get_hicp_euro_area(self, start: str | None = None, end: str | None = None, **params: Any) -> pd.DataFrame:
        """Fast helper for euro area HICP/CPI annual inflation from ECB ICP."""
        return self.get_preset("hicp_euro_area_yoy", start=start, end=end, **params)

    def get_policy_rates(self, start: str | None = None, end: str | None = None, **params: Any) -> pd.DataFrame:
        """Return ECB MRO, deposit facility and marginal lending rates."""
        frames = [
            self.get_preset("policy_rate_mro", start=start, end=end, **params),
            self.get_preset("policy_rate_deposit", start=start, end=end, **params),
            self.get_preset("policy_rate_marginal_lending", start=start, end=end, **params),
        ]
        return pd.concat(frames, ignore_index=True, sort=False)

    def get_mfi_balance_sheet(self, start: str | None = None, end: str | None = None, **params: Any) -> pd.DataFrame:
        """Return ECB BSI monetary aggregate proxies derived from MFI balance sheets."""
        frames = [
            self.get_preset("m2_notional_stock_index", start=start, end=end, **params),
            self.get_preset("m3_notional_stock_index", start=start, end=end, **params),
        ]
        return pd.concat(frames, ignore_index=True, sort=False)

    def save_series(self, df: pd.DataFrame, name: str, fmt: str = "csv", metadata: dict[str, Any] | None = None) -> Path:
        """Persist a retrieved ECB series plus sidecar metadata."""
        target = self.base_dir / f"{name}.{fmt.lower()}"
        target.parent.mkdir(parents=True, exist_ok=True)
        if fmt.lower() == "parquet":
            df.to_parquet(target, index=False)
        else:
            target = target.with_suffix(".csv")
            df.to_csv(target, index=False)
        meta = {
            "dataset": name,
            "provider": "ecb",
            "source": "ECB Data Portal SDMX 2.1 REST",
            "api_base": self.base_url,
            "rows": int(len(df)),
            "columns": list(df.columns),
            "written_at": utc_now(),
            **(metadata or {}),
        }
        target.with_suffix(".metadata.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        return target

    def sync_presets(
        self,
        presets: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
        refresh: bool = False,
        fmt: str = "csv",
        **params: Any,
    ) -> pd.DataFrame:
        """Idempotently download configured ECB presets to OfficialMacro/ECB."""
        rows: list[dict[str, Any]] = []
        preset_names = presets or [spec.name for spec in self.specs()]
        for name in preset_names:
            target = self.base_dir / f"{name}.{fmt.lower()}"
            if target.exists() and not refresh:
                rows.append({"dataset": name, "status": "cache_hit", "target_path": str(target), "rows": len(pd.read_csv(target)) if target.suffix == ".csv" else ""})
                continue
            try:
                df = self.get_preset(name, start=start, end=end, **params)
                path = self.save_series(df, name, fmt=fmt, metadata={"preset": ECB_SERIES_PRESETS.get(name, {})})
                rows.append({"dataset": name, "status": "downloaded", "target_path": str(path), "rows": len(df), "updated_at": utc_now()})
            except Exception as exc:
                LOGGER.warning("ECB preset failed name=%s error=%s", name, exc)
                rows.append({"dataset": name, "status": "failed", "error": str(exc), "updated_at": utc_now()})
        manifest = pd.DataFrame(rows)
        manifest.to_csv(self.base_dir / "ecb_official_macro_manifest.csv", index=False)
        return manifest
