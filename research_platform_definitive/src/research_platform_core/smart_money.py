"""Smart Money source catalog and lightweight coverage manifests.

Issuer-level Smart Money scoring remains in `smart_money_engine`.  This module
adds a source/coverage layer for cross-asset positioning and flow datasets such
as CFTC COT, ETF flows and options positioning.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .data_platform import resolve_data_platform_roots, utc_now


SMART_MONEY_TABLE_ROOT = Path("smart_money") / "tables"


@dataclass(frozen=True)
class SmartMoneySource:
    source_id: str
    name: str
    domain: str
    asset_classes: str
    region: str
    provider: str
    source_url: str
    status: str
    refresh_frequency: str
    notes: str


SMART_MONEY_SOURCES: tuple[SmartMoneySource, ...] = (
    SmartMoneySource(
        "cftc_cot_financial_futures",
        "CFTC COT financial futures positioning",
        "COT",
        "rates, fx, equity_index, credit",
        "usa_global",
        "CFTC public reporting",
        "https://publicreporting.cftc.gov/resource/udgc-27he.csv",
        "READY_OPTIONAL",
        "weekly",
        "Official CFTC public reporting endpoint; used for aggregate positioning where fields are available.",
    ),
    SmartMoneySource(
        "cftc_cot_legacy_futures",
        "CFTC legacy futures-only COT",
        "COT",
        "commodities, rates, fx, equity_index",
        "usa_global",
        "CFTC historical reports",
        "https://publicreporting.cftc.gov/resource/6dca-aqww.csv",
        "READY_OPTIONAL",
        "weekly",
        "Official CFTC historical/reporting page. Ingestion can use downloaded CSV/ZIP exports when API fields differ.",
    ),
    SmartMoneySource(
        "etf_flows_public_proxy",
        "ETF flows public/proxy layer",
        "ETF_FLOWS",
        "equity_etf, fixed_income_etf, commodity_etf, crypto_etf",
        "global",
        "provider_or_user_supplied",
        "",
        "PLANNED",
        "weekly",
        "Placeholder for licensed or manually supplied ETF flow files; schema and UI support are present.",
    ),
    SmartMoneySource(
        "options_positioning_proxy",
        "Options positioning and skew proxy",
        "OPTIONS",
        "equity_index, single_name, etf",
        "usa",
        "provider_or_user_supplied",
        "",
        "PLANNED",
        "daily_or_weekly",
        "Placeholder for put/call, skew and dealer positioning providers.",
    ),
    SmartMoneySource(
        "issuer_insider_fund_flows",
        "Issuer insider / fund flow evidence",
        "ISSUER_EVENTS",
        "single_name_equity",
        "usa_eu",
        "smart_money_engine",
        "",
        "PARTIAL",
        "event_driven",
        "Covered by existing smart_money_engine artifacts when source files are available locally.",
    ),
)


COT_MARKET_MAP: tuple[tuple[str, str, str], ...] = (
    ("SP500", "equity_index", "S&P"),
    ("NASDAQ", "equity_index", "NASDAQ"),
    ("US_10Y", "rates", "10-YEAR"),
    ("US_2Y", "rates", "2-YEAR"),
    ("EURO_FX", "fx", "EURO FX"),
    ("JAPANESE_YEN", "fx", "JAPANESE YEN"),
    ("BRITISH_POUND", "fx", "BRITISH POUND"),
    ("WTI_CRUDE", "commodities", "CRUDE OIL"),
    ("GOLD", "commodities", "GOLD"),
    ("COPPER", "commodities", "COPPER"),
)

DEFAULT_FLOW_ETFS: tuple[tuple[str, str, str], ...] = (
    ("SPY", "US large-cap equity", "equity"),
    ("QQQ", "US growth / Nasdaq equity", "equity"),
    ("IWM", "US small-cap equity", "equity"),
    ("GLD", "Gold ETF", "commodity"),
    ("TLT", "Long-duration Treasuries", "fixed_income"),
    ("HYG", "US high yield credit", "fixed_income"),
    ("EEM", "Emerging markets equity", "equity"),
    ("IEUR", "European equity", "equity"),
)
DEFAULT_PCR_SYMBOLS: tuple[str, ...] = ("SPY", "QQQ", "GLD")


def _read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 1:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, **kwargs)
    except Exception:
        return pd.DataFrame()


def smart_money_source_catalog() -> pd.DataFrame:
    return pd.DataFrame([asdict(source) for source in SMART_MONEY_SOURCES])


def _cot_instrument(market_name: object) -> tuple[str, str]:
    name = str(market_name or "").upper()
    for instrument, asset_class, keyword in COT_MARKET_MAP:
        if keyword in name:
            return instrument, asset_class
    return "", ""


def normalize_cot_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize CFTC COT financial/legacy rows to a common positioning schema."""
    if raw.empty:
        return pd.DataFrame()
    frame = raw.copy()
    market_col = "market_and_exchange_names" if "market_and_exchange_names" in frame.columns else "contract_market_name"
    date_col = "report_date_as_yyyy_mm_dd" if "report_date_as_yyyy_mm_dd" in frame.columns else ""
    if market_col not in frame.columns or not date_col:
        return pd.DataFrame()
    mapped = frame[market_col].map(_cot_instrument)
    frame["instrument"] = mapped.map(lambda item: item[0])
    frame["asset_class"] = mapped.map(lambda item: item[1])
    frame = frame[frame["instrument"].astype(str).str.len().gt(0)].copy()
    if frame.empty:
        return pd.DataFrame()
    frame["report_date"] = pd.to_datetime(frame[date_col], errors="coerce").dt.date.astype(str)

    if {"lev_money_positions_long", "lev_money_positions_short"}.issubset(frame.columns):
        long_col = "lev_money_positions_long"
        short_col = "lev_money_positions_short"
        trader_group = "leveraged_money"
    elif {"noncomm_positions_long_all", "noncomm_positions_short_all"}.issubset(frame.columns):
        long_col = "noncomm_positions_long_all"
        short_col = "noncomm_positions_short_all"
        trader_group = "noncommercial"
    else:
        return pd.DataFrame()

    frame["noncommercial_long"] = pd.to_numeric(frame[long_col], errors="coerce")
    frame["noncommercial_short"] = pd.to_numeric(frame[short_col], errors="coerce")
    frame["open_interest"] = pd.to_numeric(frame.get("open_interest_all", pd.Series(index=frame.index, dtype=float)), errors="coerce")
    frame["net_noncommercial"] = frame["noncommercial_long"] - frame["noncommercial_short"]
    frame["net_noncommercial_oi_pct"] = frame["net_noncommercial"] / frame["open_interest"].replace(0, float("nan"))
    frame["trader_group"] = trader_group
    out = frame[
        [
            "report_date",
            "instrument",
            "asset_class",
            market_col,
            "trader_group",
            "noncommercial_long",
            "noncommercial_short",
            "net_noncommercial",
            "net_noncommercial_oi_pct",
            "open_interest",
        ]
    ].copy()
    out = out.rename(columns={market_col: "market_name"})
    out["source"] = "CFTC COT"
    out["updated_at"] = utc_now()
    out = out.sort_values(["instrument", "report_date"]).reset_index(drop=True)
    out["weekly_change_net"] = out.groupby("instrument")["net_noncommercial"].diff()
    out["net_position_percentile_3y"] = out.groupby("instrument")["net_noncommercial"].transform(
        lambda series: series.rolling(156, min_periods=20).apply(
            lambda window: pd.Series(window).rank(pct=True).iloc[-1],
            raw=False,
        )
    )
    return out


def build_cot_hedging_pressure(cot_df: pd.DataFrame) -> pd.DataFrame:
    """Build weekly COT hedging-pressure features for ML/context panels.

    The function accepts either normalized COT rows from `normalize_cot_data()`
    or raw CFTC-style rows with commercial/non-commercial long/short columns.
    Missing commercial fields produce NaN hedging pressure rather than raising.
    """
    if cot_df is None or cot_df.empty:
        return pd.DataFrame()
    frame = cot_df.copy()
    if "instrument" not in frame.columns:
        mapped = frame.get("market_and_exchange_names", frame.get("contract_market_name", pd.Series(index=frame.index, dtype=object))).map(_cot_instrument)
        frame["instrument"] = mapped.map(lambda item: item[0])
    date_col = next((col for col in ["report_date", "report_date_as_yyyy_mm_dd", "date"] if col in frame.columns), None)
    if date_col is None:
        return pd.DataFrame()
    frame["date"] = pd.to_datetime(frame[date_col], errors="coerce")
    frame = frame.dropna(subset=["date"])
    if frame.empty:
        return pd.DataFrame()

    def pick(*names: str) -> pd.Series:
        for name in names:
            if name in frame.columns:
                return pd.to_numeric(frame[name], errors="coerce")
        return pd.Series(np.nan, index=frame.index)

    noncomm_long = pick("noncommercial_long", "noncomm_positions_long_all", "lev_money_positions_long")
    noncomm_short = pick("noncommercial_short", "noncomm_positions_short_all", "lev_money_positions_short")
    commercial_long = pick("commercial_long", "comm_positions_long_all", "commercial_positions_long_all")
    commercial_short = pick("commercial_short", "comm_positions_short_all", "commercial_positions_short_all")
    frame["net_noncommercial"] = pick("net_noncommercial").where(lambda s: s.notna(), noncomm_long - noncomm_short)
    frame["net_commercial"] = commercial_long - commercial_short
    frame["hedging_pressure"] = frame["net_commercial"] / (commercial_long + commercial_short).replace(0, np.nan)
    frame["instrument_id"] = frame["instrument"].map(lambda value: _safe_id(value).lower())

    wide_parts: list[pd.DataFrame] = []
    for instrument_id, group in frame.sort_values("date").groupby("instrument_id", dropna=True):
        if not instrument_id:
            continue
        group = group.drop_duplicates("date", keep="last").set_index("date").sort_index()
        net = pd.to_numeric(group["net_noncommercial"], errors="coerce")
        hp = pd.to_numeric(group["hedging_pressure"], errors="coerce")
        net_z = (net - net.rolling(52, min_periods=12).mean()) / net.rolling(52, min_periods=12).std().replace(0, np.nan)
        hp_z = (hp - hp.rolling(52, min_periods=12).mean()) / hp.rolling(52, min_periods=12).std().replace(0, np.nan)
        part = pd.DataFrame(
            {
                f"cot_net_noncomm_{instrument_id}": net.shift(1),
                f"cot_hedging_pressure_{instrument_id}": hp.shift(1),
                f"cot_z_{instrument_id}": hp_z.where(hp_z.notna(), net_z).shift(1),
            },
            index=group.index,
        )
        wide_parts.append(part)
    if not wide_parts:
        return pd.DataFrame()
    out = pd.concat(wide_parts, axis=1).sort_index()
    out["date"] = out.index
    return out.reset_index(drop=True)


def _safe_id(value: object) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(value).upper()).strip("_") or "UNKNOWN"


def _write_cot_contract_artifacts(history: pd.DataFrame, snapshot: pd.DataFrame, output_root: Path) -> None:
    cot_root = output_root / "smart_money" / "cot"
    cot_root.mkdir(parents=True, exist_ok=True)
    for instrument, group in history.groupby("instrument", dropna=True):
        group.to_parquet(cot_root / f"{_safe_id(instrument)}_cot_history.parquet", index=False)
    snapshot.to_csv(cot_root / "smart_money_cot_snapshot.csv", index=False)


def _history_from_frame(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    out = frame.reset_index().rename(columns={"Date": "date", "Datetime": "date", "index": "date", "Close": "close", "Adj Close": "adjclose"})
    close_col = "close" if "close" in out.columns else "adjclose" if "adjclose" in out.columns else ""
    if not close_col:
        return pd.DataFrame()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["close"] = pd.to_numeric(out[close_col], errors="coerce")
    out["symbol"] = symbol
    return out.dropna(subset=["date", "close"]).sort_values("date")[["date", "symbol", "close"]]


def _etf_descriptor(symbol: str) -> tuple[str, str]:
    lookup = {item[0]: (item[1], item[2]) for item in DEFAULT_FLOW_ETFS}
    return lookup.get(str(symbol).upper(), ("ETF flow proxy", "other"))


def refresh_cftc_cot_snapshot(
    output_root: str | Path | None = None,
    *,
    source_url: str | None = None,
    max_rows: int = 5000,
    fetch: bool = False,
) -> pd.DataFrame:
    """Fetch or load a lightweight CFTC COT snapshot.

    Network fetching is opt-in so tests and local UI startup never block.  If a
    previously saved snapshot exists it is returned when `fetch=False`.
    """
    roots = resolve_data_platform_roots(repo_output_root=output_root)
    table_root = roots.repo_output / SMART_MONEY_TABLE_ROOT
    table_root.mkdir(parents=True, exist_ok=True)
    path = table_root / "SmartMoney_COT_snapshot.csv"
    if not fetch:
        return _read_csv(path)
    return fetch_cot_data(output_root, source_url=source_url, max_rows=max_rows, fetch=True).get("snapshot", pd.DataFrame())


def fetch_cot_data(
    output_root: str | Path | None = None,
    *,
    source_url: str | None = None,
    max_rows: int = 50_000,
    fetch: bool = False,
    raw_frame: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Fetch/load and normalize CFTC COT data for core macro futures.

    Network is opt-in. Tests can pass `raw_frame`; UI can call with
    `fetch=True` when the user explicitly requests a refresh.
    """
    roots = resolve_data_platform_roots(repo_output_root=output_root)
    table_root = roots.repo_output / SMART_MONEY_TABLE_ROOT
    table_root.mkdir(parents=True, exist_ok=True)
    history_path = table_root / "SmartMoney_COT_history.csv"
    snapshot_path = table_root / "SmartMoney_COT_snapshot.csv"
    sample_path = table_root / "SmartMoney_COT_history_sample.csv"
    if raw_frame is not None:
        raw = raw_frame.copy()
    elif fetch:
        url = source_url or SMART_MONEY_SOURCES[0].source_url
        query_url = f"{url}?$limit={int(max_rows)}" if "?" not in url else url
        try:
            raw = pd.read_csv(query_url)
        except Exception as exc:
            failure = pd.DataFrame(
                [{"source_id": "cftc_cot_financial_futures", "status": "FAILED", "error": f"{type(exc).__name__}: {exc}", "updated_at": utc_now()}]
            )
            failure.to_csv(table_root / "SmartMoney_COT_fetch_status.csv", index=False)
            return {"history": _read_csv(history_path), "snapshot": _read_csv(snapshot_path), "sample": _read_csv(sample_path)}
    else:
        return {"history": _read_csv(history_path), "snapshot": _read_csv(snapshot_path), "sample": _read_csv(sample_path)}

    history = normalize_cot_data(raw)
    if history.empty:
        return {"history": pd.DataFrame(), "snapshot": pd.DataFrame(), "sample": pd.DataFrame()}
    history.to_csv(history_path, index=False)
    snapshot = history.sort_values("report_date").groupby("instrument", as_index=False).tail(1).sort_values(["asset_class", "instrument"])
    snapshot.to_csv(snapshot_path, index=False)
    sample = history.groupby("instrument", group_keys=False).tail(260).reset_index(drop=True)
    sample.to_csv(sample_path, index=False)
    _write_cot_contract_artifacts(history, snapshot, roots.repo_output)
    return {"history": history, "snapshot": snapshot, "sample": sample}


def fetch_etf_flows_proxy(
    output_root: str | Path | None = None,
    *,
    symbols: Iterable[str] | None = None,
    period: str = "1y",
    fetch: bool = False,
    history_frames: dict[str, pd.DataFrame] | None = None,
    ticker_info: dict[str, dict[str, Any]] | None = None,
) -> dict[str, pd.DataFrame]:
    """Build a public ETF-flow proxy from ETF price/AUM metadata.

    Public price data does not contain true creations/redemptions, so the output
    is labelled explicitly as a proxy. Provider-supplied flow files can replace
    this artifact later while keeping the same UI schema.
    """
    roots = resolve_data_platform_roots(repo_output_root=output_root)
    table_root = roots.repo_output / SMART_MONEY_TABLE_ROOT
    flow_root = roots.repo_output / "smart_money" / "etf_flows"
    table_root.mkdir(parents=True, exist_ok=True)
    flow_root.mkdir(parents=True, exist_ok=True)
    history_path = flow_root / "etf_flows_history.parquet"
    snapshot_path = flow_root / "etf_flows_snapshot.csv"
    table_snapshot_path = table_root / "SmartMoney_ETF_flows_snapshot.csv"
    if not fetch and history_frames is None:
        return {"history": pd.read_parquet(history_path) if history_path.exists() else pd.DataFrame(), "snapshot": _read_csv(snapshot_path)}

    if history_frames is None:
        import yfinance as yf

        history_frames = {}
        ticker_info = ticker_info or {}
        for symbol in symbols or [item[0] for item in DEFAULT_FLOW_ETFS]:
            try:
                ticker = yf.Ticker(str(symbol))
                history_frames[str(symbol).upper()] = ticker.history(period=period, interval="1d", auto_adjust=True)
                ticker_info[str(symbol).upper()] = getattr(ticker, "info", {}) or {}
            except Exception:
                history_frames[str(symbol).upper()] = pd.DataFrame()
                ticker_info[str(symbol).upper()] = {"status": "FAILED"}
    else:
        ticker_info = ticker_info or {}

    history_rows: list[pd.DataFrame] = []
    snapshot_rows: list[dict[str, Any]] = []
    for raw_symbol, frame in history_frames.items():
        symbol = str(raw_symbol).upper()
        history = _history_from_frame(frame, symbol)
        description, asset_class = _etf_descriptor(symbol)
        info = ticker_info.get(symbol, {})
        if history.empty:
            snapshot_rows.append(
                {
                    "symbol": symbol,
                    "description": description,
                    "asset_class": asset_class,
                    "data_status": "PARTIAL",
                    "error": info.get("status", "NO_DATA"),
                    "updated_at": utc_now(),
                }
            )
            continue
        total_assets = pd.to_numeric(pd.Series([info.get("totalAssets")]), errors="coerce").iloc[0]
        shares_outstanding = pd.to_numeric(pd.Series([info.get("sharesOutstanding") or info.get("sharesOutstandingImplied")]), errors="coerce").iloc[0]
        latest_close = float(history["close"].dropna().iloc[-1]) if history["close"].notna().any() else np.nan
        if pd.isna(shares_outstanding) and pd.notna(total_assets) and latest_close:
            shares_outstanding = float(total_assets) / latest_close
        history = history.copy()
        history["description"] = description
        history["asset_class"] = asset_class
        history["total_assets"] = total_assets if pd.notna(total_assets) else np.nan
        history["shares_estimate"] = shares_outstanding if pd.notna(shares_outstanding) else np.nan
        history["asset_value_proxy"] = history["close"] * history["shares_estimate"]
        for days, col in [(5, "flow_1w_proxy"), (21, "flow_1m_proxy"), (63, "flow_3m_proxy")]:
            if history["asset_value_proxy"].notna().sum() >= days + 1:
                history[col] = history["asset_value_proxy"].diff(days)
            else:
                history[col] = history["close"].pct_change(days)
        history["source"] = "yfinance_public_proxy"
        history["updated_at"] = utc_now()
        history_rows.append(history)
        latest = history.tail(1).iloc[0].to_dict()
        snapshot_rows.append(
            {
                "symbol": symbol,
                "description": description,
                "asset_class": asset_class,
                "date": pd.to_datetime(latest.get("date"), errors="coerce").date().isoformat(),
                "close": latest.get("close"),
                "total_assets": total_assets if pd.notna(total_assets) else np.nan,
                "flow_1w_proxy": latest.get("flow_1w_proxy"),
                "flow_1m_proxy": latest.get("flow_1m_proxy"),
                "flow_3m_proxy": latest.get("flow_3m_proxy"),
                "data_status": "OK",
                "source": "yfinance_public_proxy",
                "updated_at": utc_now(),
            }
        )
    out_history = pd.concat(history_rows, ignore_index=True) if history_rows else pd.DataFrame()
    snapshot = pd.DataFrame(snapshot_rows)
    if not out_history.empty:
        out_history.to_parquet(history_path, index=False)
    snapshot.to_csv(snapshot_path, index=False)
    snapshot.to_csv(table_snapshot_path, index=False)
    return {"history": out_history, "snapshot": snapshot}


def fetch_options_put_call_ratio(
    output_root: str | Path | None = None,
    *,
    symbols: Iterable[str] = DEFAULT_PCR_SYMBOLS,
    fetch: bool = False,
    option_frames: dict[str, dict[str, pd.DataFrame]] | None = None,
) -> pd.DataFrame:
    """Compute a simple put/call ratio snapshot from option-chain open interest."""
    roots = resolve_data_platform_roots(repo_output_root=output_root)
    options_root = roots.repo_output / "smart_money" / "options"
    table_root = roots.repo_output / SMART_MONEY_TABLE_ROOT
    options_root.mkdir(parents=True, exist_ok=True)
    table_root.mkdir(parents=True, exist_ok=True)
    path = options_root / "pcr_snapshot.csv"
    if not fetch and option_frames is None:
        return _read_csv(path)
    if option_frames is None:
        import yfinance as yf

        option_frames = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(str(symbol))
                expiry = ticker.options[0] if ticker.options else ""
                chain = ticker.option_chain(expiry) if expiry else None
                option_frames[str(symbol).upper()] = {
                    "calls": chain.calls if chain is not None else pd.DataFrame(),
                    "puts": chain.puts if chain is not None else pd.DataFrame(),
                }
            except Exception:
                option_frames[str(symbol).upper()] = {"calls": pd.DataFrame(), "puts": pd.DataFrame()}
    rows: list[dict[str, Any]] = []
    for raw_symbol, frames in option_frames.items():
        symbol = str(raw_symbol).upper()
        calls = frames.get("calls", pd.DataFrame())
        puts = frames.get("puts", pd.DataFrame())
        call_oi = pd.to_numeric(calls.get("openInterest", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
        put_oi = pd.to_numeric(puts.get("openInterest", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
        rows.append(
            {
                "symbol": symbol,
                "put_open_interest": float(put_oi),
                "call_open_interest": float(call_oi),
                "put_call_ratio_oi": float(put_oi / call_oi) if call_oi else np.nan,
                "data_status": "OK" if call_oi or put_oi else "PARTIAL",
                "source": "yfinance_option_chain",
                "updated_at": utc_now(),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(path, index=False)
    out.to_csv(table_root / "SmartMoney_options_pcr_snapshot.csv", index=False)
    return out


def get_pcr_cboe_bulk(
    date_range: tuple[str, str],
    output_root: str | Path | None = None,
    *,
    fetch: bool = False,
) -> pd.DataFrame:
    """Load CBOE broad put/call ratio history with a deterministic fallback.

    CBOE historical bulk formats change over time and may require manual
    download in some environments.  The function therefore prefers saved local
    history, optionally attempts a live fetch, and otherwise emits a clearly
    labelled fallback panel so downstream UI/tests never crash.
    """
    roots = resolve_data_platform_roots(repo_output_root=output_root)
    pcr_root = roots.repo_output / "smart_money" / "pcr"
    pcr_root.mkdir(parents=True, exist_ok=True)
    path = pcr_root / "cboe_pcr_history.parquet"
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    if path.exists() and not fetch:
        frame = pd.read_parquet(path)
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        return frame[frame["date"].between(start, end)].reset_index(drop=True)

    frame = pd.DataFrame()
    if fetch:
        urls = [
            "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/totalpc.csv",
            "https://www.cboe.com/us/options/market_statistics/daily/",
        ]
        for url in urls:
            try:
                candidate = pd.read_csv(url)
                if not candidate.empty:
                    frame = candidate
                    break
            except Exception:
                continue
    if not frame.empty:
        lower_cols = {col: str(col).strip().lower().replace(" ", "_") for col in frame.columns}
        frame = frame.rename(columns=lower_cols)
        date_col = next((col for col in frame.columns if "date" in col), frame.columns[0])
        frame["date"] = pd.to_datetime(frame[date_col], errors="coerce")
        numeric_cols = [col for col in frame.columns if col != "date"]
        for col in numeric_cols:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        total_col = next((col for col in frame.columns if "total" in col and ("p/c" in col or "ratio" in col or "pc" in col)), None)
        equity_col = next((col for col in frame.columns if "equity" in col and ("p/c" in col or "ratio" in col or "pc" in col)), None)
        index_col = next((col for col in frame.columns if "index" in col and ("p/c" in col or "ratio" in col or "pc" in col)), None)
        out = pd.DataFrame(
            {
                "date": frame["date"],
                "total_pcr": frame[total_col] if total_col else np.nan,
                "equity_pcr": frame[equity_col] if equity_col else np.nan,
                "index_pcr": frame[index_col] if index_col else np.nan,
                "source": "cboe_public_bulk",
                "data_status": "OK",
                "updated_at": utc_now(),
            }
        ).dropna(subset=["date"])
    else:
        dates = pd.bdate_range(start, end)
        baseline = np.linspace(0, 1, max(len(dates), 1))
        out = pd.DataFrame(
            {
                "date": dates,
                "total_pcr": 0.95 + 0.05 * np.sin(baseline * np.pi),
                "equity_pcr": 0.72 + 0.04 * np.cos(baseline * np.pi),
                "index_pcr": 1.15 + 0.06 * np.sin(baseline * 2 * np.pi),
                "source": "cboe_public_bulk_fallback",
                "data_status": "FALLBACK_ESTIMATE",
                "updated_at": utc_now(),
            }
        )
    out = out[out["date"].between(start, end)].sort_values("date").reset_index(drop=True)
    out.to_parquet(path, index=False)
    return out


def get_pcr_polygon(ticker: str, expiration_date: str, orchestrator) -> dict[str, Any] | None:
    """Compute option PCR from Polygon option snapshots when available."""
    key_getter = getattr(orchestrator, "_api_key", None)
    key = key_getter("POLYGON_API_KEY") if callable(key_getter) else ""
    if not key:
        return None
    try:
        payload = orchestrator._request_json("polygon", f"https://api.polygon.io/v3/snapshot/options/{ticker}", params={"apiKey": key})
        results = payload.get("results", []) if isinstance(payload, dict) else []
        put_oi = call_oi = 0.0
        for row in results:
            details = row.get("details", {}) if isinstance(row, dict) else {}
            if expiration_date and details.get("expiration_date") != expiration_date:
                continue
            oi = float(row.get("open_interest") or 0)
            if details.get("contract_type") == "put":
                put_oi += oi
            elif details.get("contract_type") == "call":
                call_oi += oi
        return {"symbol": ticker, "expiration_date": expiration_date, "put_call_ratio_oi": put_oi / call_oi if call_oi else np.nan, "source": "polygon_options"}
    except Exception:
        return None


def get_pcr_fmp(ticker: str, orchestrator) -> dict[str, Any] | None:
    """Fetch or proxy PCR from FMP options/volatility endpoints when available."""
    getter = getattr(orchestrator, "get_fundamentals_fmp", None)
    if not callable(getter):
        return None
    payload = getter(ticker, statement="ratios_ttm")
    if not payload:
        return None
    data = payload.get("payload", payload)
    first = data[0] if isinstance(data, list) and data else data if isinstance(data, dict) else {}
    pcr = first.get("putCallRatio") or first.get("put_call_ratio")
    try:
        pcr_value = float(pcr)
    except (TypeError, ValueError):
        pcr_value = np.nan
    return {"symbol": ticker, "put_call_ratio_oi": pcr_value, "source": "fmp_options_proxy"}


def compile_smart_money_asset_catalog(
    output_root: str | Path | None = None,
) -> pd.DataFrame:
    """Write a human-readable catalog of Smart Money v1 instruments."""
    roots = resolve_data_platform_roots(repo_output_root=output_root)
    table_root = roots.repo_output / SMART_MONEY_TABLE_ROOT
    table_root.mkdir(parents=True, exist_ok=True)
    cot_snapshot = _read_csv(table_root / "SmartMoney_COT_snapshot.csv")
    etf_snapshot = _read_csv(roots.repo_output / "smart_money" / "etf_flows" / "etf_flows_snapshot.csv")
    pcr_snapshot = _read_csv(roots.repo_output / "smart_money" / "options" / "pcr_snapshot.csv")
    rows: list[dict[str, Any]] = []
    if not cot_snapshot.empty:
        for _, row in cot_snapshot.iterrows():
            rows.append(
                {
                    "symbol": row.get("instrument", ""),
                    "description": row.get("market_name", ""),
                    "source": "CFTC COT",
                    "data_type": "cot",
                    "asset_class": row.get("asset_class", ""),
                    "status": "OK",
                    "last_updated": row.get("report_date", ""),
                }
            )
    if not etf_snapshot.empty:
        for _, row in etf_snapshot.iterrows():
            rows.append(
                {
                    "symbol": row.get("symbol", ""),
                    "description": row.get("description", ""),
                    "source": "yfinance_public_proxy",
                    "data_type": "etf_flow",
                    "asset_class": row.get("asset_class", ""),
                    "status": row.get("data_status", "PARTIAL"),
                    "last_updated": row.get("date", ""),
                }
            )
    if not pcr_snapshot.empty:
        for _, row in pcr_snapshot.iterrows():
            rows.append(
                {
                    "symbol": row.get("symbol", ""),
                    "description": "Options put/call ratio proxy",
                    "source": "yfinance_option_chain",
                    "data_type": "pcr",
                    "asset_class": "options",
                    "status": row.get("data_status", "PARTIAL"),
                    "last_updated": row.get("updated_at", ""),
                }
            )
    catalog = pd.DataFrame(rows)
    catalog.to_csv(table_root / "SmartMoneyAssetCatalog.csv", index=False)
    return catalog


def compile_smart_money_source_manifest(
    financial_db_root: str | Path | None = None,
    output_root: str | Path | None = None,
    *,
    fetch_cot: bool = False,
) -> pd.DataFrame:
    """Write source catalog + source manifest for Smart Money coverage."""
    roots = resolve_data_platform_roots(financial_db_root=financial_db_root, repo_output_root=output_root)
    table_root = roots.repo_output / SMART_MONEY_TABLE_ROOT
    table_root.mkdir(parents=True, exist_ok=True)
    catalog = smart_money_source_catalog()
    cot_bundle = fetch_cot_data(roots.repo_output, fetch=fetch_cot)
    cot = cot_bundle.get("snapshot", pd.DataFrame())
    etf_snapshot = _read_csv(roots.repo_output / "smart_money" / "etf_flows" / "etf_flows_snapshot.csv")
    pcr_snapshot = _read_csv(roots.repo_output / "smart_money" / "options" / "pcr_snapshot.csv")
    existing_coverage = _read_csv(table_root / "SmartMoney_coverage.csv")
    rows: list[dict[str, Any]] = []
    for source in SMART_MONEY_SOURCES:
        rows_count = 0
        last_date = ""
        status = source.status
        if source.source_id == "cftc_cot_financial_futures" and not cot.empty:
            rows_count = len(cot)
            status = "OK"
            date_cols = [col for col in cot.columns if "date" in col.lower()]
            if date_cols:
                dates = pd.to_datetime(cot[date_cols[0]], errors="coerce")
                if dates.notna().any():
                    last_date = dates.max().date().isoformat()
        elif source.source_id == "etf_flows_public_proxy" and not etf_snapshot.empty:
            rows_count = len(etf_snapshot)
            status = "OK" if etf_snapshot["data_status"].astype(str).str.upper().eq("OK").any() else "PARTIAL"
            if "date" in etf_snapshot.columns:
                dates = pd.to_datetime(etf_snapshot["date"], errors="coerce")
                if dates.notna().any():
                    last_date = dates.max().date().isoformat()
        elif source.source_id == "options_positioning_proxy" and not pcr_snapshot.empty:
            rows_count = len(pcr_snapshot)
            status = "OK" if pcr_snapshot["data_status"].astype(str).str.upper().eq("OK").any() else "PARTIAL"
            if "updated_at" in pcr_snapshot.columns:
                dates = pd.to_datetime(pcr_snapshot["updated_at"], errors="coerce")
                if dates.notna().any():
                    last_date = dates.max().date().isoformat()
        elif source.source_id == "issuer_insider_fund_flows" and not existing_coverage.empty:
            rows_count = int(existing_coverage.get("rows", pd.Series([0])).fillna(0).sum()) if "rows" in existing_coverage.columns else len(existing_coverage)
            status = "PARTIAL" if rows_count else "PLANNED"
        rows.append(
            {
                **asdict(source),
                "data_status": status,
                "rows": rows_count,
                "last_date": last_date,
                "manifest_updated_at": utc_now(),
            }
        )
    manifest = pd.DataFrame(rows)
    catalog.to_csv(table_root / "SmartMoneySourceCatalog.csv", index=False)
    manifest.to_csv(table_root / "SmartMoneySourceManifest.csv", index=False)
    compile_smart_money_asset_catalog(roots.repo_output)
    return manifest


def load_smart_money_source_manifest(
    financial_db_root: str | Path | None = None,
    output_root: str | Path | None = None,
    *,
    refresh_if_missing: bool = True,
) -> pd.DataFrame:
    roots = resolve_data_platform_roots(financial_db_root=financial_db_root, repo_output_root=output_root)
    path = roots.repo_output / SMART_MONEY_TABLE_ROOT / "SmartMoneySourceManifest.csv"
    frame = _read_csv(path)
    if frame.empty and refresh_if_missing:
        frame = compile_smart_money_source_manifest(roots.financial_db, roots.repo_output)
    return frame


def summarize_smart_money_sources(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["domain", "source_count", "ok_count", "ready_optional_count", "planned_count", "partial_count", "rows"])
    view = frame.copy()
    view["data_status"] = view.get("data_status", "PLANNED").fillna("PLANNED").astype(str).str.upper()
    grouped = view.groupby("domain", dropna=False)
    rows: list[dict[str, Any]] = []
    for domain, group in grouped:
        rows_series = (
            pd.to_numeric(group["rows"], errors="coerce")
            if "rows" in group.columns
            else pd.Series(0, index=group.index)
        )
        rows.append(
            {
                "domain": domain,
                "source_count": int(len(group)),
                "ok_count": int(group["data_status"].eq("OK").sum()),
                "ready_optional_count": int(group["data_status"].str.contains("READY", na=False).sum()),
                "planned_count": int(group["data_status"].str.contains("PLANNED", na=False).sum()),
                "partial_count": int(group["data_status"].str.contains("PARTIAL", na=False).sum()),
                "rows": int(rows_series.fillna(0).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("domain").reset_index(drop=True)
