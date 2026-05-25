"""End-to-end data completion orchestration for the research workstation.

The module coordinates the existing Drive-first loaders instead of introducing
new providers. Default execution is conservative: `execute=False` writes an
auditable plan/coverage manifest, while `execute=True` runs the configured
downloads with explicit safety caps.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .aqr_factors import refresh_aqr_factor_library
from .data_center_catalog import build_target_catalog, summarize_target_catalog
from .data_platform import get_ohlcv_daily_search_roots, get_ohlcv_parquet_root_info, resolve_data_platform_roots, utc_now
from .loaders.bditalia_client import BancaDItaliaClient
from .loaders.ecb_client import EcbClient
from .loaders.equity_universe import EQUITY_UNIVERSES, EquityUniverseManager
from .loaders.fama_french import FamaFrenchLoader
from .loaders.fx_commodities import ForexCommoditiesManager
from .loaders.risk_factors import RiskFactorsLoader
from .macro_features import build_credit_risk_macro_dataset, build_inflation_nowcasting_dataset, save_macro_feature_panel
from .ohlcv_ingest import OhlcvIngestJob, summarize_ohlcv_manifest
from .banking_data import run_banks_data_pipeline
from .run_lock import acquire_stage_lock, read_stage_lock, release_stage_lock

try:
    from smart_money_engine import run_smart_money_engine
except Exception:  # pragma: no cover - import style differs in app/package contexts
    from src.smart_money_engine import run_smart_money_engine


DEFAULT_UNIVERSES = ["sp500", "nasdaq100", "eurostoxx50", "ftsemib"]
DEFAULT_MARKETS = ["us_all", "europe_major", "japan_major", "global_etfs"]
DEFAULT_MACRO_OVERLAYS = ["SPY", "UUP", "TLT", "GLD", "DBC", "DX-Y.NYB"]


@dataclass(frozen=True)
class CompletionConfig:
    start_year: int = 2000
    end_year: int = 2026
    universes: tuple[str, ...] = tuple(DEFAULT_UNIVERSES)
    markets: tuple[str, ...] = tuple(DEFAULT_MARKETS)
    execute: bool = False
    incremental: bool = True
    refresh: bool = False
    max_symbols: int | None = 25
    max_assets: int | None = 250
    max_factor_datasets: int | None = 4
    include_smart_money: bool = True
    include_banking: bool = True
    include_macro: bool = True
    include_factors: bool = True

    @property
    def start_date(self) -> str:
        return f"{self.start_year}-01-01"

    @property
    def end_date(self) -> str:
        return f"{self.end_year}-12-31"


def _safe_len(value: Any) -> int:
    try:
        return int(len(value))
    except Exception:
        return 0


def _write_manifest(rows: list[dict[str, Any]], output_root: Path, name: str) -> pd.DataFrame:
    output_dir = output_root / "data_completion"
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame([{"stage": name, "status": "EMPTY", "updated_at": utc_now()}])
    frame["updated_at"] = frame.get("updated_at", utc_now())
    frame.to_csv(output_dir / f"{name}.csv", index=False)
    return frame


def _progress(message: str) -> None:
    print(f"[data_completion] {utc_now()} {message}", flush=True)


def data_source_map() -> pd.DataFrame:
    """Return the canonical source/provider map used by 2000-2026 completion."""
    rows = [
        {
            "domain": "equity_prices",
            "provider": "OhlcvIngestJob policy engine",
            "primary_sources": "yfinance; stooq fallback; configured Kaggle static seeds",
            "target": "Database Finanziario/MarketData/OHLCV",
            "history_target": "2000-2026 daily where provider history exists",
        },
        {
            "domain": "equity_fundamentals",
            "provider": "EquityUniverseManager / yfinance statements",
            "primary_sources": "Wikipedia constituents; yfinance quarterly statements",
            "target": "Database Finanziario/Equities/<region>/<universe>/fundamentals",
            "history_target": "provider-available statement history, audited as limited when not full 2000",
        },
        {
            "domain": "macro_fx_risk",
            "provider": "ForexCommoditiesManager; RiskFactorsLoader; ECB; Banca d'Italia",
            "primary_sources": "yfinance overlays; FRED CSV/API; ECB SDW; Banca d'Italia",
            "target": "Database Finanziario/FX, Commodities, RiskFactors, OfficialMacro",
            "history_target": "2000-2026 where official/provider series exists",
        },
        {
            "domain": "factor_library",
            "provider": "FamaFrenchLoader; AQRFactorProvider; derived equity factor panel",
            "primary_sources": "Ken French Data Library; AQR Data Library; local OHLCV/fundamental artifacts",
            "target": "Database Finanziario/Factors and output/ml_training_lab",
            "history_target": "2000-2026 monthly/daily depending on factor family",
        },
        {
            "domain": "smart_money",
            "provider": "smart_money_engine",
            "primary_sources": "SEC/CFTC/Treasury/USAspending/local official-source files",
            "target": "output/smart_money",
            "history_target": "best-effort official-source coverage; no fabricated events",
        },
        {
            "domain": "banking",
            "provider": "banking_data pipeline",
            "primary_sources": "ECB supervised entities; Banca d'Italia; Wikipedia; listed-bank yfinance panel",
            "target": "output/banks_pipeline",
            "history_target": "2000-2026 market panel where listed ticker history exists",
        },
    ]
    return pd.DataFrame(rows)


class ResearchDataBootstrapper:
    """Drive-first orchestration over all data families needed by the app."""

    def __init__(
        self,
        financial_db_root: str | Path | None = None,
        output_root: str | Path | None = None,
        config: CompletionConfig | None = None,
    ) -> None:
        roots = resolve_data_platform_roots(financial_db_root=financial_db_root, repo_output_root=output_root)
        self.financial_db_root = roots.financial_db
        self.output_root = roots.repo_output
        self.config = config or CompletionConfig()
        (self.output_root / "data_completion").mkdir(parents=True, exist_ok=True)

    def _status_row(self, stage: str, status: str, **extra: Any) -> dict[str, Any]:
        return {
            "stage": stage,
            "status": status,
            "execute": bool(self.config.execute),
            "start_year": self.config.start_year,
            "end_year": self.config.end_year,
            "financial_db_root": str(self.financial_db_root),
            "output_root": str(self.output_root),
            **extra,
            "updated_at": utc_now(),
        }

    def write_plan(self) -> pd.DataFrame:
        source_map = data_source_map()
        out_dir = self.output_root / "data_completion"
        source_map.to_csv(out_dir / "source_map_2000_2026.csv", index=False)
        source_map.to_csv(out_dir / "DataCompletion_source_map.csv", index=False)
        (out_dir / "bootstrap_config.json").write_text(json.dumps(asdict(self.config), indent=2, default=str), encoding="utf-8")
        return source_map

    def _stage_protected_root(self, stage: str) -> Path:
        if stage == "equity_prices":
            return get_ohlcv_parquet_root_info(self.financial_db_root, self.output_root).path
        if stage == "equity_fundamentals":
            return self.financial_db_root / "Equities"
        return self.output_root

    def _run_stage_locked(self, stage: str, func: Any) -> pd.DataFrame:
        if not self.config.execute or stage == "coverage":
            return func()
        lock = acquire_stage_lock(stage, self.output_root, protected_root=self._stage_protected_root(stage))
        if lock is None:
            existing = read_stage_lock(stage, self.output_root)
            _progress(f"stage={stage} locked running pid={getattr(existing, 'pid', '')} run_id={getattr(existing, 'run_id', '')}")
            return _write_manifest(
                [
                    self._status_row(
                        stage,
                        "LOCKED_RUNNING",
                        locked_by=getattr(existing, "run_id", ""),
                        pid=getattr(existing, "pid", ""),
                        lock_path=getattr(existing, "lock_path", ""),
                    )
                ],
                self.output_root,
                f"{stage}_2000_2026",
            )
        try:
            _progress(f"stage={stage} lock_acquired path={lock.lock_path} run_id={lock.run_id}")
            result = func()
            return result
        finally:
            release_stage_lock(stage, self.output_root, lock_info=lock)
            _progress(f"stage={stage} lock_released")

    def bootstrap_equity_fundamentals(self) -> pd.DataFrame:
        manager = EquityUniverseManager(self.financial_db_root, self.config.start_date)
        rows: list[dict[str, Any]] = []
        for universe in self.config.universes:
            _progress(f"equity_fundamentals universe={universe} start")
            if universe not in EQUITY_UNIVERSES:
                rows.append(self._status_row("equity_fundamentals", "UNKNOWN_UNIVERSE", universe=universe))
                continue
            if not self.config.execute:
                target = self.financial_db_root / "Equities" / EQUITY_UNIVERSES[universe]["region"] / universe
                rows.append(self._status_row("equity_fundamentals", "PLANNED", universe=universe, target_path=str(target)))
                continue
            try:
                constituent = manager.write_constituents(universe, refresh=self.config.refresh)
                manifest = manager.sync_fundamentals(universe, max_symbols=self.config.max_symbols, refresh=self.config.refresh)
                status = "DONE" if not str(constituent.get("status", "")).startswith("provider_failed") else "PROVIDER_LIMITED"
                rows.append(
                    self._status_row(
                        "equity_fundamentals",
                        status,
                        universe=universe,
                        rows=int(manifest.get("rows", pd.Series(dtype=float)).fillna(0).sum()) if not manifest.empty and "rows" in manifest else len(manifest),
                        target_path=str(Path(constituent.get("target_path", ""))),
                        provider_status=constituent.get("status", ""),
                        provider_error=constituent.get("error", ""),
                        note="Provider statement history may be shorter than 2000-2026; manifest preserves actual coverage.",
                    )
                )
            except Exception as exc:
                rows.append(self._status_row("equity_fundamentals", "FAILED", universe=universe, error=f"{type(exc).__name__}: {exc}"))
            _progress(f"equity_fundamentals universe={universe} done")
        return _write_manifest(rows, self.output_root, "equity_fundamentals_2000_2026")

    def bootstrap_equity_prices(self) -> pd.DataFrame:
        _progress("equity_prices start")
        parquet_info = get_ohlcv_parquet_root_info(self.financial_db_root, self.output_root)
        if not self.config.execute:
            rows = [
                self._status_row(
                    "equity_prices",
                    "PLANNED",
                    markets=",".join(self.config.markets),
                    mode="incremental" if self.config.incremental else "full",
                    target_path=str(self.financial_db_root / "MarketData" / "OHLCV"),
                    parquet_root=str(parquet_info.path),
                    parquet_root_source=parquet_info.source,
                    parquet_storage_mode=parquet_info.storage_mode,
                )
            ]
            return _write_manifest(rows, self.output_root, "equity_prices_2000_2026")
        job = OhlcvIngestJob(self.financial_db_root, self.output_root)
        _progress(f"equity_prices parquet_root={job.parquet_root} mode={job.parquet_storage_mode}")
        manifest = job.run_daily(
            markets=list(self.config.markets),
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            mode="incremental" if self.config.incremental else "full",
            max_assets=self.config.max_assets,
            batch_size=80,
            dry_run=False,
        )
        summary = summarize_ohlcv_manifest(manifest)
        rows = []
        for row in summary.to_dict("records"):
            row = dict(row)
            price_status = row.pop("status", "")
            rows.append(self._status_row("equity_prices", "DONE", price_status=price_status, **row))
        _progress("equity_prices done")
        return _write_manifest(rows, self.output_root, "equity_prices_2000_2026")

    def bootstrap_macro_fx_series(self) -> pd.DataFrame:
        _progress("macro_fx start")
        rows: list[dict[str, Any]] = []
        if not self.config.include_macro:
            return _write_manifest([self._status_row("macro_fx", "SKIPPED")], self.output_root, "macro_fx_2000_2026")
        if not self.config.execute:
            return _write_manifest(
                [self._status_row("macro_fx", "PLANNED", overlays=",".join(DEFAULT_MACRO_OVERLAYS), target_path=str(self.financial_db_root))],
                self.output_root,
                "macro_fx_2000_2026",
            )
        fx = ForexCommoditiesManager(self.financial_db_root, self.config.start_date)
        risk = RiskFactorsLoader(self.financial_db_root, self.config.start_date)
        rows.extend(fx.sync_fx(refresh=self.config.refresh, max_symbols=None).assign(stage="macro_fx", family="fx").to_dict("records"))
        rows.extend(fx.sync_commodities(refresh=self.config.refresh, max_symbols=None).assign(stage="macro_fx", family="commodities").to_dict("records"))
        rows.extend(risk.sync_fred(refresh=self.config.refresh).assign(stage="macro_fx", family="fred_risk").to_dict("records"))
        rows.extend(risk.sync_volatility(refresh=self.config.refresh).assign(stage="macro_fx", family="volatility").to_dict("records"))
        risk_panel = risk.build_derived_risk_factors()
        rows.append(self._status_row("macro_fx", "DONE", family="risk_panel", rows=len(risk_panel), target_path=str(self.financial_db_root / "RiskFactors" / "risk_factor_panel.csv")))
        ecb = EcbClient(self.financial_db_root, self.output_root)
        bdi = BancaDItaliaClient(self.financial_db_root, self.output_root)
        ecb_manifest = ecb.sync_presets(["hicp_euro_area_yoy", "policy_rate_mro", "policy_rate_deposit", "m2_notional_stock_index", "m3_notional_stock_index"], start=self.config.start_date, end=self.config.end_date, refresh=self.config.refresh)
        bdi_manifest = bdi.sync_presets(["credit_growth", "deposit_volumes", "public_debt", "official_rates_bdi"], start=self.config.start_date, end=self.config.end_date, refresh=self.config.refresh)
        rows.extend(ecb_manifest.assign(stage="macro_fx", family="ecb").to_dict("records"))
        rows.extend(bdi_manifest.assign(stage="macro_fx", family="bditalia").to_dict("records"))
        feature_dir = self.financial_db_root / "OfficialMacro" / "features"
        inflation = build_inflation_nowcasting_dataset(ecb_client=ecb, bditalia_client=bdi, start=self.config.start_date)
        credit = build_credit_risk_macro_dataset(ecb_client=ecb, bditalia_client=bdi, start=self.config.start_date)
        save_macro_feature_panel(inflation, feature_dir / "inflation_nowcasting_panel.csv", fmt="csv")
        save_macro_feature_panel(credit, feature_dir / "credit_risk_macro_panel.csv", fmt="csv")
        rows.append(self._status_row("macro_fx", "DONE", family="official_macro_features", rows=len(inflation) + len(credit), target_path=str(feature_dir)))
        _progress("macro_fx done")
        return _write_manifest(rows, self.output_root, "macro_fx_2000_2026")

    def bootstrap_factor_libraries(self) -> pd.DataFrame:
        _progress("factor_libraries start")
        if not self.config.include_factors:
            return _write_manifest([self._status_row("factor_libraries", "SKIPPED")], self.output_root, "factor_libraries_2000_2026")
        if not self.config.execute:
            return _write_manifest([self._status_row("factor_libraries", "PLANNED", target_path=str(self.financial_db_root / "Factors"))], self.output_root, "factor_libraries_2000_2026")
        rows: list[dict[str, Any]] = []
        ff = FamaFrenchLoader(self.financial_db_root, self.output_root).download_all(refresh=self.config.refresh, max_datasets=self.config.max_factor_datasets)
        rows.extend(ff.assign(stage="factor_libraries", family="fama_french").to_dict("records"))
        aqr_paths = refresh_aqr_factor_library(self.financial_db_root, self.output_root, refresh=self.config.refresh, max_datasets=self.config.max_factor_datasets)
        rows.append(self._status_row("factor_libraries", "DONE", family="aqr", target_path=json.dumps({k: str(v) for k, v in aqr_paths.items()})))
        _progress("factor_libraries done")
        return _write_manifest(rows, self.output_root, "factor_libraries_2000_2026")

    def bootstrap_smart_money_data(self) -> pd.DataFrame:
        _progress("smart_money start")
        if not self.config.include_smart_money:
            return _write_manifest([self._status_row("smart_money", "SKIPPED")], self.output_root, "smart_money_2000_2026")
        target = self.output_root / "smart_money"
        if not self.config.execute:
            return _write_manifest([self._status_row("smart_money", "PLANNED", target_path=str(target))], self.output_root, "smart_money_2000_2026")
        result = run_smart_money_engine(self.financial_db_root, target, max_files_per_source=50)
        rows = [self._status_row("smart_money", "DONE", target_path=str(target), manifest=json.dumps(result.get("manifest", {}), default=str))]
        _progress("smart_money done")
        return _write_manifest(rows, self.output_root, "smart_money_2000_2026")

    def bootstrap_banking_universe(self) -> pd.DataFrame:
        _progress("banking start")
        if not self.config.include_banking:
            return _write_manifest([self._status_row("banking", "SKIPPED")], self.output_root, "banking_2000_2026")
        target = self.output_root / "banks_pipeline"
        if not self.config.execute:
            return _write_manifest([self._status_row("banking", "PLANNED", target_path=str(target))], self.output_root, "banking_2000_2026")
        artifacts = run_banks_data_pipeline(
            output_dir=target,
            cache_dir=target / "_cache",
            market_start=self.config.start_date,
            include_market=True,
            include_ecb_macro=True,
        )
        rows = [self._status_row("banking", "DONE", target_path=str(target), rows=json.dumps({name: _safe_len(df) for name, df in artifacts.items()}))]
        _progress("banking done")
        return _write_manifest(rows, self.output_root, "banking_2000_2026")

    def build_factor_universe_panel(self, max_symbols: int | None = None) -> pd.DataFrame:
        """Build a point-in-time-safe factor panel from local OHLCV parquet files."""
        from ml_stock_lab.factor_registry import add_factor_scores
        from research_platform_core.macro_context import add_macro_context_features

        _progress("factor_universe_panel start")
        max_symbols = self.config.max_assets if max_symbols is None else max_symbols
        ohlcv_roots = get_ohlcv_daily_search_roots(self.financial_db_root, self.output_root)
        seen_roots: set[Path] = set()
        candidates = []
        for root in ohlcv_roots:
            if root in seen_roots:
                continue
            seen_roots.add(root)
            candidates.extend(root.glob("*/*.parquet"))
        candidates += list((self.financial_db_root / "Equities").glob("*/*/prices/*.parquet"))
        if max_symbols:
            candidates = candidates[: int(max_symbols)]
        frames: list[pd.DataFrame] = []
        failures: list[dict[str, Any]] = []
        total_candidates = len(candidates)
        for idx, path in enumerate(candidates, start=1):
            if idx == 1 or idx % 250 == 0 or idx == total_candidates:
                _progress(f"factor_universe_panel progress files={idx}/{total_candidates} frames={len(frames)} failures={len(failures)} latest={path.name}")
            try:
                df = pd.read_parquet(path, columns=["date", "adjclose"])
                close_col = "adjclose"
            except Exception:
                try:
                    df = pd.read_parquet(path, columns=["date", "close"])
                    close_col = "close"
                except Exception as exc:
                    failures.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
                    continue
            try:
                if df.empty or "date" not in df.columns:
                    continue
                ticker = path.stem.replace("_", ".").upper()
                if close_col not in df.columns:
                    continue
                view = df[["date", close_col]].copy()
                view["date"] = pd.to_datetime(view["date"], errors="coerce")
                view = view.dropna(subset=["date"]).sort_values("date")
                view["ticker"] = ticker
                price = pd.to_numeric(view[close_col], errors="coerce")
                view["market_value"] = price
                view["price"] = price
                view["ret21d"] = price.pct_change(21)
                view["ret63d"] = price.pct_change(63)
                view["ret126d"] = price.pct_change(126)
                view["ret252d"] = price.pct_change(252)
                returns = price.pct_change()
                view["vol63d"] = returns.rolling(63).std() * np.sqrt(252)
                view["vol126d"] = returns.rolling(126).std() * np.sqrt(252)
                view["vol252d"] = returns.rolling(252).std() * np.sqrt(252)
                view["forward_return_21d"] = price.shift(-21) / price - 1
                view["forward_return_63d"] = price.shift(-63) / price - 1
                view["forward_return_252d"] = price.shift(-252) / price - 1
                view["forward_return"] = view["forward_return_21d"]
                frames.append(
                    view[
                        [
                            "date",
                            "ticker",
                            "market_value",
                            "price",
                            "ret21d",
                            "ret63d",
                            "ret126d",
                            "ret252d",
                            "vol63d",
                            "vol126d",
                            "vol252d",
                            "forward_return",
                            "forward_return_21d",
                            "forward_return_63d",
                            "forward_return_252d",
                        ]
                    ]
                )
            except Exception as exc:
                failures.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
                continue
        panel = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame(columns=["date", "ticker", "market_value"])
        if not panel.empty:
            panel = panel[(panel["date"].dt.year >= self.config.start_year) & (panel["date"].dt.year <= self.config.end_year)].copy()
            panel = add_factor_scores(panel)
            panel = add_macro_context_features(panel, self.financial_db_root, self.output_root)
            panel["target_horizon_days"] = 21
            panel["data_start_year"] = self.config.start_year
            panel["data_end_year"] = self.config.end_year
            panel["provenance"] = "local_ohlcv_parquet"
        target_out = self.output_root / "ml_training_lab" / "tables" / "FactorUniversePanel.csv"
        target_db = self.financial_db_root / "Factors" / "EquityFactorPanel.csv"
        target_out.parent.mkdir(parents=True, exist_ok=True)
        target_db.parent.mkdir(parents=True, exist_ok=True)
        if failures:
            failure_path = self.output_root / "ml_training_lab" / "tables" / "FactorUniversePanel_failures.csv"
            failure_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(failures).to_csv(failure_path, index=False)
            _progress(f"factor_universe_panel failures={len(failures)} path={failure_path}")
        panel.to_csv(target_out, index=False)
        try:
            panel.to_csv(target_db, index=False)
        except OSError as exc:
            _progress(f"factor_universe_panel drive_write_failed error={type(exc).__name__}: {exc}")
        _write_manifest(
            [self._status_row("factor_universe_panel", "DONE" if not panel.empty else "EMPTY", rows=len(panel), tickers=panel["ticker"].nunique() if "ticker" in panel else 0, target_path=str(target_out))],
            self.output_root,
            "factor_universe_panel_2000_2026",
        )
        _progress("factor_universe_panel done")
        return panel

    def write_coverage_report(self) -> pd.DataFrame:
        _progress("coverage_report start")
        catalog = build_target_catalog(self.financial_db_root, self.output_root)
        summary = summarize_target_catalog(catalog)
        out = self.output_root / "data_completion"
        out.mkdir(parents=True, exist_ok=True)
        catalog.to_csv(out / "DataCompletion_target_catalog.csv", index=False)
        summary.to_csv(out / "DataCompletion_target_summary.csv", index=False)
        quick_contract = {
            "generated_at": utc_now(),
            "financial_db_root": str(self.financial_db_root),
            "mode": "targeted_data_completion_coverage",
            "note": "Backfill coverage uses the canonical target catalog instead of a recursive Drive scan, so cloud placeholder files cannot stall the run.",
            "target_catalog": str(out / "DataCompletion_target_catalog.csv"),
            "target_summary": str(out / "DataCompletion_target_summary.csv"),
            "domains": summary.to_dict(orient="records"),
        }
        (out / "DataCompletion_quick_coverage_contract.json").write_text(json.dumps(quick_contract, indent=2, default=str), encoding="utf-8")
        _progress("coverage_report done")
        return summary

    def run_all(self) -> dict[str, pd.DataFrame]:
        _progress("run_all start")
        self.write_plan()
        results: dict[str, pd.DataFrame] = {}
        steps = [
            ("equity_fundamentals", self.bootstrap_equity_fundamentals),
            ("equity_prices", self.bootstrap_equity_prices),
            ("macro_fx", self.bootstrap_macro_fx_series),
            ("factor_libraries", self.bootstrap_factor_libraries),
            ("smart_money", self.bootstrap_smart_money_data),
            ("banking", self.bootstrap_banking_universe),
            ("factor_universe_panel", self.build_factor_universe_panel),
            ("coverage", self.write_coverage_report),
        ]
        for name, func in steps:
            _progress(f"stage={name} start")
            results[name] = self._run_stage_locked(name, func)
            _progress(f"stage={name} rows={len(results[name])} done")
        manifest = {
            "generated_at": utc_now(),
            "config": asdict(self.config),
            "outputs": {name: {"rows": len(frame)} for name, frame in results.items()},
        }
        (self.output_root / "data_completion" / "full_completion_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        _progress("run_all complete")
        return results

    def run_stage(self, stage: str) -> pd.DataFrame:
        stage_map = {
            "equity_fundamentals": self.bootstrap_equity_fundamentals,
            "equity_prices": self.bootstrap_equity_prices,
            "macro_fx": self.bootstrap_macro_fx_series,
            "factor_libraries": self.bootstrap_factor_libraries,
            "smart_money": self.bootstrap_smart_money_data,
            "banking": self.bootstrap_banking_universe,
            "factor_universe_panel": self.build_factor_universe_panel,
            "coverage": self.write_coverage_report,
        }
        if stage not in stage_map:
            raise ValueError(f"Unknown data completion stage: {stage}")
        return self._run_stage_locked(stage, stage_map[stage])


def run_full_data_completion(
    financial_db_root: str | Path | None = None,
    output_root: str | Path | None = None,
    **kwargs: Any,
) -> dict[str, pd.DataFrame]:
    config = CompletionConfig(**kwargs)
    return ResearchDataBootstrapper(financial_db_root, output_root, config).run_all()


def _bootstrapper(
    financial_db_root: str | Path | None = None,
    output_root: str | Path | None = None,
    **kwargs: Any,
) -> ResearchDataBootstrapper:
    return ResearchDataBootstrapper(financial_db_root, output_root, CompletionConfig(**kwargs))


def bootstrap_equity_fundamentals(
    start_year: int = 2000,
    end_year: int = 2026,
    financial_db_root: str | Path | None = None,
    output_root: str | Path | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Bootstrap equity constituents/fundamentals with the existing provider policy."""
    return _bootstrapper(financial_db_root, output_root, start_year=start_year, end_year=end_year, **kwargs).bootstrap_equity_fundamentals()


def bootstrap_equity_prices(
    start_year: int = 2000,
    end_year: int = 2026,
    financial_db_root: str | Path | None = None,
    output_root: str | Path | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Bootstrap OHLCV prices for configured markets, 2000-2026 by default."""
    return _bootstrapper(financial_db_root, output_root, start_year=start_year, end_year=end_year, **kwargs).bootstrap_equity_prices()


def bootstrap_macro_fx_series(
    start_year: int = 2000,
    end_year: int = 2026,
    financial_db_root: str | Path | None = None,
    output_root: str | Path | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Bootstrap macro, FX, rates, volatility and overlay series."""
    return _bootstrapper(financial_db_root, output_root, start_year=start_year, end_year=end_year, **kwargs).bootstrap_macro_fx_series()


def bootstrap_smart_money_data(
    start_year: int = 2000,
    end_year: int = 2026,
    financial_db_root: str | Path | None = None,
    output_root: str | Path | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Bootstrap Smart Money / government-data artifacts with local official-source policy."""
    return _bootstrapper(financial_db_root, output_root, start_year=start_year, end_year=end_year, **kwargs).bootstrap_smart_money_data()


def bootstrap_banking_universe(
    start_year: int = 2000,
    end_year: int = 2026,
    financial_db_root: str | Path | None = None,
    output_root: str | Path | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Bootstrap Banking Data Lab universe and optional market/macro panels."""
    return _bootstrapper(financial_db_root, output_root, start_year=start_year, end_year=end_year, **kwargs).bootstrap_banking_universe()


def build_factor_universe_panel(
    start_year: int = 2000,
    end_year: int = 2026,
    financial_db_root: str | Path | None = None,
    output_root: str | Path | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Build the local price-derived factor panel consumed by the ML training lab."""
    return _bootstrapper(financial_db_root, output_root, start_year=start_year, end_year=end_year, **kwargs).build_factor_universe_panel()


def validate_completion_coverage(
    financial_db_root: str | Path | None = None,
    output_root: str | Path | None = None,
    start_year: int = 2000,
    end_year: int = 2026,
    strict: bool = False,
) -> pd.DataFrame:
    """Validate whether completion artifacts cover the requested historical range.

    The validator is intentionally transparent: provider-limited domains can be
    `PLANNED`/`EMPTY` in non-strict mode for local smoke tests, but full release
    checks can pass `strict=True` to require completed artifacts and date range
    evidence where the manifest exposes `min_date`/`max_date`.
    """
    roots = resolve_data_platform_roots(financial_db_root=financial_db_root, repo_output_root=output_root)
    out_dir = roots.repo_output / "data_completion"
    parquet_info = get_ohlcv_parquet_root_info(roots.financial_db, roots.repo_output)
    parquet_files = list((parquet_info.path / "daily").glob("*/*.parquet")) if (parquet_info.path / "daily").exists() else []
    failures_path = roots.repo_output / "tables" / "OHLCV_write_failures.csv"
    provider_failures_path = roots.repo_output / "tables" / "OHLCV_provider_failures.csv"
    failure_count = 0
    provider_failure_count = 0
    critical_provider_failure_count = 0
    if failures_path.exists() and failures_path.stat().st_size > 1:
        try:
            failure_count = len(pd.read_csv(failures_path))
        except Exception:
            failure_count = 0
    if provider_failures_path.exists() and provider_failures_path.stat().st_size > 1:
        try:
            provider_failures = pd.read_csv(provider_failures_path)
            provider_failure_count = len(provider_failures)
            critical_types = {"NETWORK_TIMEOUT", "PROVIDER_ERROR", "INVALID_SYMBOL", "WRITE_FAILED"}
            critical_provider_failure_count = int(provider_failures.get("error_type", pd.Series(dtype=object)).fillna("").astype(str).str.upper().isin(critical_types).sum())
        except Exception:
            provider_failure_count = 0
            critical_provider_failure_count = 0
    expected = {
        "equity_fundamentals": out_dir / "equity_fundamentals_2000_2026.csv",
        "equity_prices": out_dir / "equity_prices_2000_2026.csv",
        "macro_fx": out_dir / "macro_fx_2000_2026.csv",
        "factor_libraries": out_dir / "factor_libraries_2000_2026.csv",
        "smart_money": out_dir / "smart_money_2000_2026.csv",
        "banking": out_dir / "banking_2000_2026.csv",
        "factor_universe_panel": out_dir / "factor_universe_panel_2000_2026.csv",
    }
    rows: list[dict[str, Any]] = []
    for domain, path in expected.items():
        exists = path.exists() and path.stat().st_size > 0
        status = "MISSING"
        row_count = 0
        min_date = pd.NA
        max_date = pd.NA
        if exists:
            try:
                frame = pd.read_csv(path)
                row_count = len(frame)
                status_values = set(frame.get("status", pd.Series(dtype=object)).astype(str))
                if "DONE" in status_values or "OK" in status_values:
                    status = "OK"
                elif "PLANNED" in status_values and not strict:
                    status = "PLANNED"
                elif "SKIPPED" in status_values and not strict:
                    status = "SKIPPED"
                elif "EMPTY" in status_values and not strict:
                    status = "EMPTY"
                else:
                    status = ";".join(sorted(status_values)) or "UNKNOWN"
                date_cols = [c for c in frame.columns if c.lower() in {"date", "min_date", "max_date", "start_date", "end_date"}]
                date_values = pd.concat([pd.to_datetime(frame[c], errors="coerce") for c in date_cols], ignore_index=True) if date_cols else pd.Series(dtype="datetime64[ns]")
                if date_values.notna().any():
                    min_date = date_values.min().date().isoformat()
                    max_date = date_values.max().date().isoformat()
            except Exception as exc:
                status = f"INVALID:{type(exc).__name__}"
        passes = bool(exists and status in {"OK", "PLANNED", "SKIPPED", "EMPTY"} and (not strict or status == "OK"))
        if strict and pd.notna(min_date) and pd.notna(max_date):
            passes = passes and int(str(min_date)[:4]) <= start_year and int(str(max_date)[:4]) >= end_year
        extra: dict[str, Any] = {}
        if domain == "equity_prices":
            extra = {
                "ohlcv_parquet_root": str(parquet_info.path),
                "ohlcv_parquet_root_source": parquet_info.source,
                "ohlcv_parquet_storage_mode": parquet_info.storage_mode,
                "ohlcv_parquet_files": len(parquet_files),
                "ohlcv_write_failures": failure_count,
                "ohlcv_write_failures_path": str(failures_path),
                "ohlcv_provider_failures": provider_failure_count,
                "ohlcv_critical_provider_failures": critical_provider_failure_count,
                "ohlcv_provider_failures_path": str(provider_failures_path),
            }
            if strict and (failure_count > 0 or critical_provider_failure_count > 0):
                passes = False
        rows.append(
            {
                "domain": domain,
                "status": status,
                "passes": passes,
                "strict": bool(strict),
                "rows": row_count,
                "min_date": min_date,
                "max_date": max_date,
                "manifest_path": str(path),
                **extra,
            }
        )
    report = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(out_dir / "DataCompletion_coverage_validation.csv", index=False)
    return report
