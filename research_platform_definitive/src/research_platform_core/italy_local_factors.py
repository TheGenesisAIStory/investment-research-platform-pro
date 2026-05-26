"""Italy-specific local factor and macro context helpers.

The module is intentionally restart-friendly: public API fetches are optional,
and deterministic proxy artifacts are emitted when live endpoints are
unavailable.  This keeps the Italy layer visible in the workstation while
preserving a clear ``PARTIAL`` status until official fundamentals/regulatory
feeds are connected.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from .data_platform import resolve_data_platform_roots, utc_now


ITALY_LOCAL_UNIVERSE: tuple[str, ...] = (
    "ISP.MI",
    "UCG.MI",
    "ENEL.MI",
    "ENI.MI",
    "STM.MI",
    "BMPS.MI",
    "MB.MI",
    "G.MI",
    "TIT.MI",
    "LDO.MI",
    "AMP.MI",
    "RACE.MI",
    "BPER.MI",
    "CPR.MI",
    "PIRC.MI",
)


def _artifact_root(output_root: str | Path | None = None) -> Path:
    roots = resolve_data_platform_roots(repo_output_root=output_root)
    path = roots.repo_output / "italy_factors"
    path.mkdir(parents=True, exist_ok=True)
    return path


def fetch_ecb_sovereign_spread(output_root: str | Path | None = None, *, fetch: bool = False) -> pd.DataFrame:
    """Fetch or synthesize an ECB-style BTP-Bund spread panel."""
    root = _artifact_root(output_root)
    path = root / "ecb_btp_bund_spread.parquet"
    if path.exists() and not fetch:
        return pd.read_parquet(path)
    frame = pd.DataFrame()
    if fetch:
        try:
            import requests

            urls = {
                "DE_10Y": "https://data-api.ecb.europa.eu/service/data/YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y?format=csvdata",
                "IT_10Y": "https://data-api.ecb.europa.eu/service/data/YC/B.IT.EUR.4F.G_N_A.SV_C_YM.SR_10Y?format=csvdata",
            }
            series = {}
            for key, url in urls.items():
                response = requests.get(url, timeout=30, headers={"User-Agent": "Gen.is.IA ItalyFactors/1.0"})
                response.raise_for_status()
                raw = pd.read_csv(StringIO(response.text))
                date_col = next((col for col in raw.columns if "TIME_PERIOD" in col.upper()), None)
                value_col = next((col for col in raw.columns if col.upper() in {"OBS_VALUE", "VALUE"}), None)
                if date_col and value_col:
                    s = raw[[date_col, value_col]].copy()
                    s["date"] = pd.to_datetime(s[date_col], errors="coerce")
                    s[key] = pd.to_numeric(s[value_col], errors="coerce")
                    series[key] = s.set_index("date")[key]
            if series:
                frame = pd.concat(series, axis=1).reset_index()
                if {"IT_10Y", "DE_10Y"}.issubset(frame.columns):
                    frame["btp_bund_10y"] = frame["IT_10Y"] - frame["DE_10Y"]
        except Exception:
            frame = pd.DataFrame()
    if frame.empty:
        dates = pd.date_range("2005-01-31", pd.Timestamp.today().normalize(), freq="ME")
        cycle = np.linspace(0, 8 * np.pi, len(dates))
        frame = pd.DataFrame(
            {
                "date": dates,
                "DE_10Y": 1.5 + 0.8 * np.sin(cycle / 3),
                "IT_10Y": 2.5 + 1.1 * np.sin(cycle / 3) + 0.4 * np.sin(cycle),
                "btp_bund_10y": 1.0 + 0.3 * np.sin(cycle),
                "data_status": "PARTIAL_PROXY",
                "source": "ecb_proxy_fallback",
                "updated_at": utc_now(),
            }
        )
    else:
        frame["data_status"] = "OK"
        frame["source"] = "ecb_sdw"
        frame["updated_at"] = utc_now()
    frame.to_parquet(path, index=False)
    return frame


def fetch_eurostat_macro(output_root: str | Path | None = None, *, fetch: bool = False) -> pd.DataFrame:
    """Fetch or synthesize Eurostat macro context for Italy/EU."""
    root = _artifact_root(output_root)
    path = root / "eurostat_macro_context.parquet"
    if path.exists() and not fetch:
        return pd.read_parquet(path)
    # Live Eurostat parsing is intentionally deferred to a dedicated connector;
    # the fallback keeps the schema stable for UI and tests.
    dates = pd.date_range("2005-03-31", pd.Timestamp.today().normalize(), freq="QE")
    cycle = np.linspace(0, 6 * np.pi, len(dates))
    frame = pd.DataFrame(
        {
            "date": dates,
            "it_gdp_yoy": 0.01 + 0.02 * np.sin(cycle),
            "euro_area_gdp_yoy": 0.012 + 0.015 * np.sin(cycle + 0.4),
            "it_hicp_yoy": 0.02 + 0.012 * np.cos(cycle / 2),
            "it_unemployment": 0.09 + 0.015 * np.sin(cycle / 2),
            "data_status": "PARTIAL_PROXY" if not fetch else "PARTIAL",
            "source": "eurostat_schema_fallback",
            "updated_at": utc_now(),
        }
    )
    frame.to_parquet(path, index=False)
    return frame


def build_italy_factor_panel(start_date: str = "2005-01-01", output_root: str | Path | None = None, *, fetch: bool = False) -> pd.DataFrame:
    """Build a lightweight Italy market/factor context panel.

    Live OHLCV fetch is optional.  Without API data, the function writes a
    documented proxy panel with market, BTP-Bund and banking-context columns.
    """
    root = _artifact_root(output_root)
    dates = pd.bdate_range(pd.to_datetime(start_date), pd.Timestamp.today().normalize())
    rows: list[pd.DataFrame] = []
    if fetch:
        try:
            import yfinance as yf

            for ticker in ITALY_LOCAL_UNIVERSE:
                hist = yf.download(ticker, start=start_date, auto_adjust=True, progress=False)
                if hist is None or hist.empty:
                    continue
                close = pd.to_numeric(hist["Close"], errors="coerce")
                frame = pd.DataFrame({"date": close.index, "ticker": ticker, "price": close.values})
                rows.append(frame)
        except Exception:
            rows = []
    if rows:
        panel = pd.concat(rows, ignore_index=True)
        panel["ret_21d"] = panel.sort_values(["ticker", "date"]).groupby("ticker")["price"].pct_change(21)
        panel["momentum_12m_1m"] = panel.sort_values(["ticker", "date"]).groupby("ticker")["price"].pct_change(252) - panel.groupby("ticker")["price"].pct_change(21)
        panel["data_status"] = "PARTIAL"
        panel["source"] = "yfinance_italy"
    else:
        base = pd.DataFrame({"date": dates})
        cycle = np.linspace(0, 10 * np.pi, len(base))
        panel = pd.concat(
            [
                base.assign(
                    ticker=ticker,
                    price=100 * (1 + 0.0002 * np.arange(len(base)) + 0.05 * np.sin(cycle + idx / 3)),
                    ret_21d=lambda x: x["price"].pct_change(21),
                    momentum_12m_1m=lambda x: x["price"].pct_change(252) - x["price"].pct_change(21),
                    data_status="PARTIAL_PROXY",
                    source="italy_factor_schema_fallback",
                )
                for idx, ticker in enumerate(ITALY_LOCAL_UNIVERSE)
            ],
            ignore_index=True,
        )
    spread = fetch_ecb_sovereign_spread(output_root, fetch=False)
    if not spread.empty:
        spread_frame = spread[["date", "btp_bund_10y"]].copy()
        spread_frame["date"] = pd.to_datetime(spread_frame["date"], errors="coerce")
        panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
        panel = pd.merge_asof(
            panel.sort_values("date"),
            spread_frame.dropna(subset=["date"]).sort_values("date"),
            on="date",
            direction="backward",
        )
    panel["updated_at"] = utc_now()
    panel.to_parquet(root / "italy_factor_panel.parquet", index=False)
    pd.DataFrame(
        [
            {
                "domain": "italy_local_factors",
                "status": "PARTIAL" if panel["data_status"].astype(str).str.contains("PARTIAL").any() else "OK",
                "tickers": len(ITALY_LOCAL_UNIVERSE),
                "rows": len(panel),
                "first_date": str(pd.to_datetime(panel["date"]).min().date()),
                "last_date": str(pd.to_datetime(panel["date"]).max().date()),
                "updated_at": utc_now(),
            }
        ]
    ).to_csv(root / "italy_factor_manifest.csv", index=False)
    return panel


__all__ = [
    "ITALY_LOCAL_UNIVERSE",
    "build_italy_factor_panel",
    "fetch_ecb_sovereign_spread",
    "fetch_eurostat_macro",
]
