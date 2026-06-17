"""Repeatable ML training pipeline for the 2000-2026 research lab.

The module is deliberately artifact-first: it loads the factor panel produced
by `research_platform_core.data_completion` when available, falls back to the
Financial DB/artifact panel, trains time-split expected-return models, and
writes auditable model cards plus predictions.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .datasets import load_financial_db_panel, normalize_panel, validate_panel_coverage
from .evaluation import oos_r2, rank_information_coefficient, rolling_ic_by_date, sharpe_ratio
from .features import add_basic_features, make_forward_returns, select_numeric_features
from .factor_registry import FACTOR_BLOCKS, available_factor_blocks, feature_columns_for_blocks
from .macro_features import build_macro_context_features
from .prediction import ExpectedReturnModel, describe_temporal_split

try:
    from research_platform_core.llm_client import OllamaClient
    from research_platform_core.alpha101 import Alpha101Suite
    from research_platform_core.macro_context import load_macro_context_panel
except Exception:  # pragma: no cover - optional dependency during isolated package use
    OllamaClient = None
    Alpha101Suite = None
    load_macro_context_panel = None


DEFAULT_MODELS = ("ols", "rf")


def _add_alpha101_features_if_available(panel: pd.DataFrame, feature_blocks: tuple[str, ...]) -> pd.DataFrame:
    if "alpha101" not in feature_blocks or Alpha101Suite is None or panel.empty:
        return panel
    if not {"date", "ticker"}.issubset(panel.columns):
        return panel
    frame = panel.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    has_real_ohlcv = {"open", "high", "low", "close", "volume"}.issubset(frame.columns)
    if "close" not in frame.columns and "price" in frame.columns:
        frame["close"] = pd.to_numeric(frame["price"], errors="coerce")
    for col in ["open", "high", "low"]:
        if col not in frame.columns and "close" in frame.columns:
            frame[col] = pd.to_numeric(frame["close"], errors="coerce")
    if "volume" not in frame.columns:
        if {"market_value", "close"}.issubset(frame.columns):
            close = pd.to_numeric(frame["close"], errors="coerce").replace(0, np.nan)
            frame["volume"] = pd.to_numeric(frame["market_value"], errors="coerce") / close
        else:
            frame["volume"] = 1.0
    required = {"date", "ticker", "open", "high", "low", "close", "volume"}
    if not required.issubset(frame.columns):
        return panel
    if not has_real_ohlcv:
        return _add_alpha101_proxy_features(frame)
    pivots = {}
    for col in ["open", "high", "low", "close", "volume"]:
        pivots[col] = frame.pivot_table(index="date", columns="ticker", values=col, aggfunc="last").sort_index()
    vwap = frame.pivot_table(index="date", columns="ticker", values="vwap", aggfunc="last").sort_index() if "vwap" in frame.columns else (pivots["high"] + pivots["low"] + pivots["close"]) / 3
    returns = pivots["close"].pct_change()
    suite = Alpha101Suite()
    results = suite.compute_all(
        close=pivots["close"],
        open_=pivots["open"],
        high=pivots["high"],
        low=pivots["low"],
        volume=pivots["volume"],
        vwap=vwap,
        returns=returns,
    )
    alpha_panel = suite.to_flat_panel(results).reset_index()
    alpha_panel["date"] = pd.to_datetime(alpha_panel["date"], errors="coerce")
    return frame.merge(alpha_panel, on=["date", "ticker"], how="left")


def _add_alpha101_proxy_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Fast fallback when the factor panel has price but not full OHLCV.

    The full Alpha101 formulas need open/high/low/close/volume.  Some historical
    factor panels only carry ``price`` and ``market_value``; for those panels we
    expose lightweight cross-sectional rank proxies so the experimental
    `alpha101` block can be benchmarked without blocking the training job.
    """
    out = frame.sort_values(["ticker", "date"]).copy()
    close = pd.to_numeric(out["close"], errors="coerce")
    out["_alpha_ret1"] = close.groupby(out["ticker"]).pct_change(1)
    out["_alpha_ret5"] = close.groupby(out["ticker"]).pct_change(5)
    out["_alpha_ret21"] = close.groupby(out["ticker"]).pct_change(21)
    out["_alpha_ret63"] = close.groupby(out["ticker"]).pct_change(63)
    out["_alpha_vol21"] = out["_alpha_ret1"].groupby(out["ticker"]).rolling(21, min_periods=5).std().reset_index(level=0, drop=True)
    out["_alpha_size"] = pd.to_numeric(out.get("market_value"), errors="coerce") if "market_value" in out.columns else close
    source_cols = ["_alpha_ret1", "_alpha_ret5", "_alpha_ret21", "_alpha_ret63", "_alpha_vol21", "_alpha_size"]
    alpha_columns: dict[str, pd.Series] = {}
    for idx in range(1, 102):
        source = source_cols[(idx - 1) % len(source_cols)]
        values = pd.to_numeric(out[source], errors="coerce")
        if idx % 3 == 0:
            values = -values
        alpha_columns[f"alpha{idx:03d}"] = values.groupby(out["date"]).rank(pct=True)
    out = out.drop(columns=[col for col in out.columns if col.startswith("_alpha_")])
    return pd.concat([out, pd.DataFrame(alpha_columns, index=out.index)], axis=1).sort_index()


def _add_macro_context_features_if_available(
    panel: pd.DataFrame,
    feature_blocks: tuple[str, ...],
    output_root: str | Path,
    financial_db_root: str | Path | None = None,
) -> pd.DataFrame:
    if not {"macro_context", "macro_regime"}.intersection(feature_blocks) or panel.empty or load_macro_context_panel is None:
        return panel
    try:
        macro = load_macro_context_panel(financial_db_root=financial_db_root, output_root=output_root, refresh_if_missing=True)
    except Exception:
        macro = pd.DataFrame()
    if macro.empty:
        return panel
    try:
        return build_macro_context_features(panel, macro)
    except Exception:
        return panel


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_csv(path: Path, max_rows: int | None = None) -> pd.DataFrame:
    try:
        if max_rows and max_rows > 0 and path.stat().st_size > 50_000_000:
            # Large factor panels are ordered by source file/ticker. A plain
            # head(max_rows) can collapse the temporal validation window, so
            # bounded smoke runs sample across the whole artifact in chunks.
            chunks: list[pd.DataFrame] = []
            per_chunk = max(250, int(max_rows) // 40)
            random_state = 1729
            for chunk_idx, chunk in enumerate(pd.read_csv(path, chunksize=100_000)):
                if chunk.empty:
                    continue
                take = min(len(chunk), per_chunk)
                if len(chunk) > take:
                    chunk = chunk.sample(n=take, random_state=random_state + chunk_idx)
                chunks.append(chunk)
            if not chunks:
                return pd.DataFrame()
            sampled = pd.concat(chunks, ignore_index=True, sort=False)
            if len(sampled) > int(max_rows):
                sampled = sampled.sample(n=int(max_rows), random_state=random_state)
            return sampled
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _write_csv(df: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return str(path)


def _write_json(payload: dict[str, Any], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return str(path)


def _candidate_factor_panels(output_root: Path, financial_db_root: Path | None) -> list[Path]:
    project_root = _project_root()
    candidates = [
        output_root / "ml_training_lab" / "tables" / "FactorUniversePanel.csv",
        output_root / "tables" / "FactorUniversePanel.csv",
        project_root / "output" / "ml_training_lab" / "tables" / "FactorUniversePanel.csv",
    ]
    if financial_db_root is not None:
        candidates.extend(
            [
                financial_db_root / "Factors" / "EquityFactorPanel.csv",
                financial_db_root / "Factors" / "equity_factor_panel.csv",
            ]
        )
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in candidates:
        expanded = path.expanduser()
        if expanded not in seen:
            seen.add(expanded)
            unique.append(expanded)
    return unique


def load_training_panel(
    output_root: str | Path,
    financial_db_root: str | Path | None = None,
    start_year: int = 2000,
    end_year: int = 2026,
    max_rows: int | None = None,
    target_horizon_days: int = 21,
) -> pd.DataFrame:
    """Load and normalize the best available 2000-2026 training panel."""
    output_root = Path(output_root)
    db_root = Path(financial_db_root).expanduser() if financial_db_root else None
    panel = pd.DataFrame()
    for path in _candidate_factor_panels(output_root, db_root):
        if path.exists() and path.stat().st_size > 1:
            panel = _read_csv(path, max_rows=max_rows)
            if not panel.empty:
                panel["source_artifact"] = str(path)
                break
    if panel.empty:
        panel = load_financial_db_panel(financial_db_root=db_root, output_root=output_root)
    panel = normalize_panel(panel)
    if not panel.empty and "date" in panel.columns:
        dates = pd.to_datetime(panel["date"], errors="coerce")
        panel = panel[(dates.dt.year >= int(start_year)) & (dates.dt.year <= int(end_year))].copy()
    if not panel.empty:
        panel = add_basic_features(panel)
        price_col = "price" if "price" in panel.columns else "market_value"
        if "forward_return" not in panel.columns or pd.to_numeric(panel["forward_return"], errors="coerce").notna().sum() == 0:
            panel = make_forward_returns(panel, price_col=price_col, horizon=target_horizon_days)
        if f"forward_return_{int(target_horizon_days)}d" in panel.columns:
            panel["forward_return"] = pd.to_numeric(panel[f"forward_return_{int(target_horizon_days)}d"], errors="coerce")
        panel["target_horizon_days"] = int(target_horizon_days)
        if "date" in panel.columns and "ticker" in panel.columns:
            panel = panel.sort_values(["date", "ticker"])
    if max_rows and max_rows > 0 and len(panel) > int(max_rows):
        panel = panel.head(int(max_rows))
    return panel.reset_index(drop=True)


def _split_by_year(panel: pd.DataFrame, train_end_year: int, test_start_year: int) -> tuple[pd.Index, pd.Index, dict[str, Any]]:
    if "date" not in panel.columns:
        cutoff = int(len(panel) * 0.75)
        train_idx = panel.index[:cutoff]
        test_idx = panel.index[cutoff:]
        return train_idx, test_idx, {
            "split_type": "row_order",
            "split_criterion": f"row_position < {cutoff}; row_position >= {cutoff}",
            "train_rows": len(train_idx),
            "test_rows": len(test_idx),
        }
    dates = pd.to_datetime(panel["date"], errors="coerce")
    train_idx = panel[dates.dt.year <= int(train_end_year)].index
    test_idx = panel[dates.dt.year >= int(test_start_year)].index
    if len(train_idx) == 0 or len(test_idx) == 0:
        cutoff = int(len(panel) * 0.75)
        train_idx = panel.index[:cutoff]
        test_idx = panel.index[cutoff:]
    return train_idx, test_idx, describe_temporal_split(panel, train_idx, test_idx)


def _feature_importance(model: ExpectedReturnModel, feature_names: list[str]) -> pd.DataFrame:
    estimator = getattr(model, "estimator_", None)
    values: Iterable[float] | None = None
    if estimator is not None and hasattr(estimator, "feature_importances_"):
        values = getattr(estimator, "feature_importances_")
    elif estimator is not None and hasattr(estimator, "coef_"):
        values = np.ravel(getattr(estimator, "coef_"))
    elif hasattr(model, "beta_"):
        values = np.ravel(getattr(model, "beta_"))[1:]
    if values is None:
        return pd.DataFrame({"feature": feature_names, "importance": np.nan, "direction": ""})
    rows = []
    for feature, value in zip(feature_names, values):
        numeric = float(value) if pd.notna(value) else np.nan
        rows.append({"feature": feature, "importance": numeric, "direction": "positive" if numeric >= 0 else "negative"})
    return pd.DataFrame(rows).sort_values("importance", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


def _long_short_from_predictions(predictions: pd.DataFrame) -> pd.Series:
    if predictions.empty or not {"date", "expected_return", "forward_return"}.issubset(predictions.columns):
        return pd.Series(dtype=float, name="long_short_return")
    frame = predictions.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["rank"] = frame.groupby("date")["expected_return"].rank(pct=True)
    long_ret = frame[frame["rank"] >= 0.8].groupby("date")["forward_return"].mean()
    short_ret = frame[frame["rank"] <= 0.2].groupby("date")["forward_return"].mean()
    return (long_ret - short_ret).dropna().rename("long_short_return")


def _fit_index_for_model(train_idx: pd.Index, model_name: str, feature_blocks: tuple[str, ...], max_fit_rows: int = 15_000) -> pd.Index:
    """Bound expensive tree fits for local Alpha101 workstation runs."""
    if "alpha101" not in feature_blocks or str(model_name) not in {"rf", "gbrt", "ensemble"}:
        return train_idx
    if len(train_idx) <= int(max_fit_rows):
        return train_idx
    return pd.Index(pd.Series(train_idx).sample(n=int(max_fit_rows), random_state=1729).sort_values().to_numpy())


def _prediction_wide(predictions: pd.DataFrame, models: Iterable[str]) -> pd.DataFrame:
    if predictions.empty or not {"date", "ticker", "model", "expected_return", "expected_return_rank"}.issubset(predictions.columns):
        return pd.DataFrame()
    keys = [col for col in ["date", "ticker", "market_value", "price", "forward_return"] if col in predictions.columns]
    base = predictions[keys].drop_duplicates(["date", "ticker"], keep="first") if {"date", "ticker"}.issubset(predictions.columns) else pd.DataFrame()
    wide = base.copy()
    for model_name in list(models):
        view = predictions[predictions["model"].astype(str).eq(str(model_name))]
        if view.empty:
            continue
        cols = ["date", "ticker", "expected_return", "expected_return_rank"]
        pivot = view[[c for c in cols if c in view.columns]].copy()
        pivot = pivot.rename(
            columns={
                "expected_return": f"expected_return_{model_name}",
                "expected_return_rank": f"expected_return_rank_{model_name}",
            }
        )
        pivot[f"score_{model_name}"] = pd.to_numeric(pivot.get(f"expected_return_rank_{model_name}"), errors="coerce") * 100
        wide = wide.merge(pivot, on=["date", "ticker"], how="outer") if not wide.empty else pivot
    score_cols = [col for col in wide.columns if col.startswith("score_")]
    if score_cols:
        wide["score_composite"] = wide[score_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
        wide["ml_score"] = wide["score_composite"]
        wide["ml_quintile"] = np.ceil((wide["ml_score"].rank(pct=True).fillna(0) * 5).clip(1, 5)).astype("Int64")
    return wide


def summarize_training_with_ollama(
    metrics: pd.DataFrame,
    model: str = "llama3.1",
    base_url: str = "http://localhost:11434",
    timeout: int = 30,
) -> dict[str, Any]:
    """Ask local Ollama for a compact research summary; fail soft when offline."""
    if metrics.empty:
        prompt_metrics = "No metrics available."
    else:
        prompt_metrics = metrics.to_string(index=False)
    prompt = (
        "Sei un revisore quantitativo buy-side. Riassumi in italiano i risultati "
        "del training ML, evidenziando metriche, rischi di overfitting e prossimi controlli.\n\n"
        f"{prompt_metrics}"
    )
    if OllamaClient is None:
        return {"status": "UNAVAILABLE", "model": model, "summary": "", "reason": "OllamaClient unavailable"}
    response = OllamaClient(base_url=base_url, model=model, timeout=timeout).generate_completion(prompt)
    return {"status": response.status, "model": response.model, "summary": response.content, "reason": response.error}


def train_ml_model_suite(
    output_root: str | Path,
    financial_db_root: str | Path | None = None,
    start_year: int = 2000,
    end_year: int = 2026,
    train_end_year: int = 2018,
    test_start_year: int = 2019,
    models: Iterable[str] = DEFAULT_MODELS,
    max_rows: int | None = None,
    use_ollama: bool = False,
    ollama_model: str = "llama3.1",
    target_horizon_days: int = 21,
    feature_blocks: Iterable[str] | None = None,
    cost_bps: float = 10.0,
) -> dict[str, Any]:
    """Train expected-return models and persist v1.0 research artifacts."""
    output_root = Path(output_root).expanduser()
    lab_root = output_root / "ml_training_lab"
    tables = lab_root / "tables"
    model_dir = lab_root / "models"
    tables.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    models = tuple(models)
    feature_blocks = tuple(feature_blocks or FACTOR_BLOCKS.keys())
    panel = load_training_panel(output_root, financial_db_root, start_year, end_year, max_rows=max_rows, target_horizon_days=target_horizon_days)
    panel = _add_macro_context_features_if_available(panel, feature_blocks, output_root=output_root, financial_db_root=financial_db_root)
    panel = _add_alpha101_features_if_available(panel, feature_blocks)
    panel_path = _write_csv(panel, tables / "MLTraining_panel.csv")
    coverage = validate_panel_coverage(panel, min_tickers=2, min_dates=2, required_columns=("market_value",))
    _write_csv(coverage, tables / "MLTraining_coverage.csv")

    min_non_null = max(3, min(25, len(panel) // 10))
    feature_cols = select_numeric_features(panel, target="forward_return", min_non_null=min_non_null, feature_blocks=list(feature_blocks), include_extra_numeric=False) if not panel.empty else []
    if not feature_cols and not panel.empty:
        feature_cols = select_numeric_features(panel, target="forward_return", min_non_null=min_non_null, feature_blocks=list(feature_blocks), include_extra_numeric=True)
    feature_cols = [c for c in feature_cols if c not in {"market_value", "price", "open", "high", "low", "close", "volume", "vwap"}]
    target = pd.to_numeric(panel.get("forward_return", pd.Series(dtype=float)), errors="coerce") if not panel.empty else pd.Series(dtype=float)

    metrics_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []
    model_cards: list[dict[str, Any]] = []

    train_idx, test_idx, split_meta = _split_by_year(panel, train_end_year, test_start_year) if not panel.empty else (pd.Index([]), pd.Index([]), {})
    can_train = bool(not panel.empty and feature_cols and len(train_idx) > 5 and len(test_idx) > 0 and target.loc[train_idx].notna().sum() > 5)

    for model_name in list(models):
        row: dict[str, Any] = {
            "model": model_name,
            "status": "SKIPPED",
            "reason": "",
            "start_year": int(start_year),
            "end_year": int(end_year),
            "train_end_year": int(train_end_year),
            "test_start_year": int(test_start_year),
            "panel_rows": len(panel),
            "feature_count": len(feature_cols),
            "features": ",".join(feature_cols),
            **split_meta,
            "r2_os": pd.NA,
            "sharpe_long_short": pd.NA,
            "prediction_rows": 0,
            "fit_rows": pd.NA,
            "target": "forward_return",
            "target_horizon_days": int(target_horizon_days),
            "feature_blocks": ",".join(feature_blocks),
            "available_factor_blocks": ",".join(available_factor_blocks(panel)),
            "ic": pd.NA,
            "rank_ic": pd.NA,
            "sharpe_long_short_net_cost": pd.NA,
        }
        if not can_train:
            row["reason"] = "INSUFFICIENT_PANEL_OR_FEATURES"
            metrics_rows.append(row)
            continue
        try:
            X = panel[feature_cols].replace([np.inf, -np.inf], np.nan)
            fit_idx = _fit_index_for_model(train_idx, model_name, feature_blocks)
            model_kwargs = {}
            if "alpha101" in feature_blocks and str(model_name) == "rf":
                model_kwargs = {"n_estimators": 20}
            elif "alpha101" in feature_blocks and str(model_name) in {"gbrt", "ensemble"}:
                model_kwargs = {"n_estimators": 50, "rf_estimators": 20, "gbrt_estimators": 50}
            model = ExpectedReturnModel(model=model_name, **model_kwargs).fit(X.loc[fit_idx], target.loc[fit_idx])
            preds = model.predict(X.loc[test_idx])
            pred_frame = panel.loc[test_idx, [c for c in ["date", "ticker", "market_value", "price", "forward_return"] if c in panel.columns]].copy()
            pred_frame["model"] = model_name
            pred_frame["expected_return"] = preds.values
            pred_frame["expected_return_rank"] = pred_frame.groupby("date")["expected_return"].rank(pct=True) if "date" in pred_frame else pred_frame["expected_return"].rank(pct=True)
            prediction_frames.append(pred_frame)
            long_short = _long_short_from_predictions(pred_frame)
            net_long_short = (long_short - (float(cost_bps) / 10000.0) * 2.0).rename("net_long_short_return")
            ic_frame = rolling_ic_by_date(pred_frame, score_col="expected_return", return_col="forward_return")
            importances = _feature_importance(model, feature_cols)
            importances["model"] = model_name
            importance_frames.append(importances)
            model_path = model_dir / f"expected_return_{model_name}_{start_year}_{end_year}.pkl"
            with model_path.open("wb") as handle:
                pickle.dump(model, handle)
            model_card = {
                "model": model_name,
                "model_path": str(model_path),
                "training_window": f"{start_year}-{train_end_year}",
                "test_window": f"{test_start_year}-{end_year}",
                "features": feature_cols,
                "target": "forward_return",
                "target_horizon_days": int(target_horizon_days),
                "feature_blocks": list(feature_blocks),
                "panel_path": panel_path,
                "split": split_meta,
                "metrics": {
                    "r2_os": oos_r2(target.loc[test_idx], preds),
                    "rank_ic": rank_information_coefficient(pred_frame["forward_return"], pred_frame["expected_return"]),
                    "sharpe_long_short": sharpe_ratio(long_short, periods_per_year=12 if len(long_short) < 80 else 252),
                },
            }
            model_cards.append(model_card)
            row.update(
                {
                    "status": "OK",
                    "reason": "OK",
                    "r2_os": oos_r2(target.loc[test_idx], preds),
                    "ic": float(ic_frame["ic"].mean()) if not ic_frame.empty else pd.NA,
                    "rank_ic": float(ic_frame["rank_ic"].mean()) if not ic_frame.empty else pd.NA,
                    "sharpe_long_short": sharpe_ratio(long_short, periods_per_year=12 if len(long_short) < 80 else 252),
                    "sharpe_long_short_net_cost": sharpe_ratio(net_long_short, periods_per_year=12 if len(net_long_short) < 80 else 252),
                    "prediction_rows": len(pred_frame),
                    "model_path": str(model_path),
                    "fit_rows": len(fit_idx),
                }
            )
        except Exception as exc:
            row["status"] = "FAILED"
            row["reason"] = f"{type(exc).__name__}: {exc}"
        metrics_rows.append(row)

    metrics = pd.DataFrame(metrics_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True, sort=False) if prediction_frames else pd.DataFrame()
    predictions_wide = _prediction_wide(predictions, models)
    importance = pd.concat(importance_frames, ignore_index=True, sort=False) if importance_frames else pd.DataFrame()
    if predictions.empty:
        predictions = pd.DataFrame(columns=["date", "ticker", "market_value", "price", "forward_return", "model", "expected_return", "expected_return_rank"])
    if importance.empty:
        importance = pd.DataFrame(columns=["model", "feature", "importance", "direction"])

    paths = {
        "panel": panel_path,
        "coverage": _write_csv(coverage, tables / "MLTraining_coverage.csv"),
        "metrics": _write_csv(metrics, tables / "MLTraining_metrics.csv"),
        "predictions": _write_csv(predictions, tables / "MLTraining_predictions.csv"),
        "predictions_wide": _write_csv(predictions_wide, tables / "MLTraining_predictions_wide.csv"),
        "feature_importance": _write_csv(importance, tables / "MLTraining_feature_importance.csv"),
        "model_cards": _write_csv(pd.DataFrame(model_cards), tables / "MLTraining_model_cards.csv"),
        "metadata": _write_json(
            {
                "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
                "start_year": start_year,
                "end_year": end_year,
                "train_end_year": train_end_year,
                "test_start_year": test_start_year,
                "models": list(models),
                "target": "forward_return",
                "target_horizon_days": int(target_horizon_days),
                "feature_blocks": list(feature_blocks),
                "leakage_policy": "factor_registry_allowlist_plus_denylist",
                "model_cards": model_cards,
            },
            lab_root / "MLTraining_manifest.json",
        ),
    }

    ml_stock_tables = output_root / "ml_stock_lab" / "tables"
    has_ok_model = bool(not metrics.empty and metrics.get("status", pd.Series(dtype=object)).astype(str).eq("OK").any())
    if has_ok_model:
        paths["ml_stock_lab_model_comparison"] = _write_csv(metrics, ml_stock_tables / "MLStockLab_model_comparison.csv")
    if not predictions.empty:
        signals = predictions_wide.copy() if not predictions_wide.empty else predictions.copy()
        if "score_composite" in signals.columns:
            signals["score"] = signals["score_composite"]
            signals["ml_score"] = signals["score_composite"]
        else:
            signals["score"] = signals.get("expected_return", pd.Series(dtype=float))
            signals["ml_score"] = pd.to_numeric(signals.get("expected_return_rank", pd.Series(dtype=float)), errors="coerce") * 100
        if len(signals) >= 5:
            signals["quintile"] = np.ceil((pd.to_numeric(signals["ml_score"], errors="coerce").rank(pct=True).fillna(0) * 5).clip(1, 5)).astype("Int64")
            signals["ml_quintile"] = "Q" + signals["quintile"].astype(str)
        else:
            signals["quintile"] = pd.NA
            signals["ml_quintile"] = pd.NA
        paths["ml_stock_lab_trained_signals"] = _write_csv(signals, ml_stock_tables / "MLStockLab_trained_model_signals.csv")

    ollama_summary = summarize_training_with_ollama(metrics, model=ollama_model) if use_ollama else {"status": "SKIPPED", "summary": ""}
    _write_json(ollama_summary, lab_root / "MLTraining_ollama_summary.json")
    paths["ollama_summary"] = str(lab_root / "MLTraining_ollama_summary.json")

    status = "OK" if not metrics.empty and metrics["status"].eq("OK").any() else "NO_TRAINED_MODELS"
    return {
        "status": status,
        "paths": paths,
        "panel": panel,
        "metrics": metrics,
        "predictions": predictions,
        "predictions_wide": predictions_wide,
        "feature_importance": importance,
        "coverage": coverage,
        "ollama_summary": ollama_summary,
    }
