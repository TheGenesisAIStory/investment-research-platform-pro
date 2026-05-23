"""High-level experiment runner for notebook/app artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .datasets import FundamentalDatasetBuilder, load_aqr_factor_panel, load_financial_db_panel, validate_panel_coverage
from .evaluation import evaluate_quintile_backtest, oos_r2, sharpe_ratio
from .features import add_basic_features, make_forward_returns, select_numeric_features
from .portfolio import make_quantile_portfolios
from .prediction import ExpectedReturnModel, describe_temporal_split, temporal_train_test_split
from .screening import rank_scores, top_bottom
from .signals import compute_relative_mispricing, cross_sectional_zscore
from .valuation import PeerImpliedValuator


def _write(df: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return str(path)


def _csv_value(values: list[str] | tuple[str, ...] | None) -> str:
    return ",".join(str(v) for v in values or [])


def _candidate_model_factor_paths(output_root: Path, financial_db_root: str | Path | None = None) -> list[Path]:
    workspace = output_root
    for _ in range(4):
        workspace = workspace.parent
    candidates = [
        output_root / "tables",
        output_root.parent / "tables",
        output_root.parent / "company_valuation" / "output" / "tables",
        output_root.parent.parent / "company_valuation" / "output" / "tables",
        workspace / "company_valuation" / "output" / "tables",
        workspace / "output" / "tables",
        Path.cwd() / "company_valuation" / "output" / "tables",
        Path.cwd() / "output" / "tables",
    ]
    if financial_db_root is not None:
        candidates.extend([
            Path(financial_db_root) / "artifacts" / "company_valuation" / "tables",
            Path(financial_db_root) / "company_valuation" / "output" / "tables",
        ])
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in candidates:
        resolved = path.expanduser()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def _load_model_based_factors(output_root: Path, financial_db_root: str | Path | None = None) -> pd.DataFrame:
    """Load valuation model factors produced by company valuation artifacts."""
    filenames = [
        "DCFModelBasedFactors.csv",
        "ResidualIncomeModelBasedFactors.csv",
        "EVAModelBasedFactors.csv",
        "ModelBasedFactors.csv",
    ]
    frames: list[pd.DataFrame] = []
    for folder in _candidate_model_factor_paths(output_root, financial_db_root):
        for filename in filenames:
            path = folder / filename
            if not path.exists():
                continue
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            if df.empty or "ticker" not in df.columns:
                continue
            df = df.copy()
            df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    merged = frames[0]
    for frame in frames[1:]:
        add_cols = [c for c in frame.columns if c != "ticker" and c not in merged.columns]
        if add_cols:
            merged = merged.merge(frame[["ticker", *add_cols]], on="ticker", how="outer")
    return merged


def _merge_model_based_factors(panel: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    if panel.empty or factors.empty or "ticker" not in panel.columns or "ticker" not in factors.columns:
        return panel
    out = panel.copy()
    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
    factor_cols = [c for c in factors.columns if c != "ticker" and c not in out.columns]
    if not factor_cols:
        return out
    return out.merge(factors[["ticker", *factor_cols]], on="ticker", how="left")


def _merge_date_factor_panel(panel: pd.DataFrame, factor_panel: pd.DataFrame, prefix: str = "aqr") -> pd.DataFrame:
    if panel.empty or factor_panel.empty or "date" not in panel.columns:
        return panel
    out = panel.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    factors = factor_panel.copy()
    if "date" in factors.columns:
        factors["date"] = pd.to_datetime(factors["date"], errors="coerce")
    else:
        factors = factors.reset_index().rename(columns={factors.index.name or "index": "date"})
        factors["date"] = pd.to_datetime(factors["date"], errors="coerce")
    factors = factors.dropna(subset=["date"]).sort_values("date")
    if factors.empty or out["date"].notna().sum() == 0:
        return out
    rename = {c: f"{prefix}_{c}" for c in factors.columns if c != "date" and c not in out.columns}
    factors = factors.rename(columns=rename)
    try:
        merged = pd.merge_asof(
            out.sort_values("date"),
            factors.sort_values("date"),
            on="date",
            direction="backward",
        )
        return merged.sort_index()
    except Exception:
        return out


def run_ml_stock_lab_experiment(
    output_root: str | Path,
    financial_db_root: str | Path | None = None,
    model: str = "ols",
    max_rows: int = 2000,
    target: str = "market_value",
    feature_blocks: list[str] | None = None,
    universe: str | None = None,
    tickers: list[str] | None = None,
    min_tickers: int = 5,
    min_dates: int = 2,
    include_aqr: bool = False,
    aqr_slugs: list[str] | None = None,
    aqr_max_datasets: int | None = None,
) -> dict[str, Any]:
    """Run a lightweight ML valuation/screening/quintile experiment."""
    output_root = Path(output_root)
    tables = output_root / "tables"
    figures = output_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    panel = load_financial_db_panel(financial_db_root=financial_db_root, identifiers=tickers)
    panel = add_basic_features(panel)
    model_factors = _load_model_based_factors(output_root, financial_db_root)
    panel = _merge_model_based_factors(panel, model_factors).head(max_rows)
    aqr_panel = pd.DataFrame()
    if include_aqr:
        aqr_panel = load_aqr_factor_panel(
            slugs=aqr_slugs,
            refresh=False,
            max_datasets=aqr_max_datasets,
            financial_db_root=financial_db_root,
            output_root=output_root.parent if output_root.name == "ml_stock_lab" else output_root,
        )
        panel = _merge_date_factor_panel(panel, aqr_panel).head(max_rows)
    proxy_source = "market_value"
    if "market_value_proxy_source" in panel.columns and panel["market_value_proxy_source"].notna().any():
        proxy_source = str(panel["market_value_proxy_source"].dropna().iloc[0])
    panel["target_source"] = proxy_source
    if "forward_return" not in panel.columns:
        panel = make_forward_returns(panel, price_col="price" if "price" in panel.columns else "market_value")

    panel_status = validate_panel_coverage(panel, min_tickers=min_tickers, min_dates=min_dates)
    panel_status["stage"] = "panel_validation"
    panel_status["model"] = model
    panel_status["target_selected"] = target
    panel_status["feature_blocks"] = _csv_value(feature_blocks)
    panel_status["universe"] = universe or ""
    panel_status["tickers"] = _csv_value(tickers)
    panel_status["model_based_factor_columns"] = _csv_value([c for c in model_factors.columns if c != "ticker"]) if not model_factors.empty else ""
    panel_status["aqr_factor_columns"] = _csv_value(list(aqr_panel.columns)) if not aqr_panel.empty else ""
    if not bool(panel_status["should_run"].iloc[0]):
        empty = pd.DataFrame()
        status = str(panel_status["status"].iloc[0])
        reason = str(panel_status["reason"].iloc[0])
        metrics = pd.DataFrame([{
            "status": status,
            "reason": reason,
            "model": model,
            "rows": 0,
            "features": "",
            "feature_count": 0,
            "feature_blocks": _csv_value(feature_blocks),
            "universe": universe or "",
            "tickers": _csv_value(tickers),
            "target": "market_value",
            "target_selected": target,
            "target_source": proxy_source,
            "panel_rows": int(panel_status["panel_rows"].iloc[0]),
            "ticker_count": int(panel_status["ticker_count"].iloc[0]),
            "date_count": int(panel_status["date_count"].iloc[0]),
            "min_date": panel_status["min_date"].iloc[0],
            "max_date": panel_status["max_date"].iloc[0],
            "r2_os": pd.NA,
            "sharpe_long_short": pd.NA,
            "volatility_long_short": pd.NA,
            "avg_names_long_short": pd.NA,
        }])
        prediction_metrics = pd.DataFrame([{
            "status": "SKIPPED",
            "reason": reason,
            "expected_return_model": "rf",
            "r2_os": pd.NA,
            "test_rows": 0,
        }])
        paths = {
            "panel": _write(panel, tables / "MLStockLab_panel.csv"),
            "signals": _write(empty, tables / "MLStockLab_signals.csv"),
            "top": _write(empty, tables / "MLStockLab_top.csv"),
            "bottom": _write(empty, tables / "MLStockLab_bottom.csv"),
            "quintiles": _write(empty, tables / "MLStockLab_quintile_returns.csv"),
            "quintile_metrics": _write(empty, tables / "MLStockLab_quintile_metrics.csv"),
            "prediction_metrics": _write(prediction_metrics, tables / "MLStockLab_prediction_metrics.csv"),
            "long_short_diagnostics": _write(empty, tables / "MLStockLab_long_short_diagnostics.csv"),
            "status": _write(panel_status, tables / "MLStockLab_status.csv"),
            "metrics": _write(metrics, tables / "MLStockLab_metrics.csv"),
        }
        return {"status": status, "paths": paths, "panel": panel, "metrics": metrics, "status_report": panel_status}

    features = select_numeric_features(panel, target="market_value", min_non_null=max(3, min(10, len(panel) // 10)))
    if not features:
        numeric = panel.select_dtypes("number").columns.tolist()
        features = [c for c in numeric if c not in {"market_value", "forward_return"}][:8]
    X, y, features = FundamentalDatasetBuilder(features, "market_value").build(panel)
    if X.empty or not features:
        empty = pd.DataFrame()
        reason = "NO_MODEL_FEATURES" if not features else "NO_TRAINABLE_ROWS"
        metrics = pd.DataFrame([{
            "status": "NO_SIGNALS",
            "reason": reason,
            "model": model,
            "rows": 0,
            "features": _csv_value(features),
            "feature_count": len(features),
            "feature_blocks": _csv_value(feature_blocks),
            "universe": universe or "",
            "tickers": _csv_value(tickers),
            "target": "market_value",
            "target_selected": target,
            "target_source": proxy_source,
            "r2_os": pd.NA,
            "sharpe_long_short": pd.NA,
            "volatility_long_short": pd.NA,
            "avg_names_long_short": pd.NA,
        }])
        prediction_metrics = pd.DataFrame([{
            "status": "SKIPPED",
            "reason": reason,
            "expected_return_model": "rf",
            "r2_os": pd.NA,
            "test_rows": 0,
        }])
        paths = {
            "panel": _write(panel, tables / "MLStockLab_panel.csv"),
            "signals": _write(empty, tables / "MLStockLab_signals.csv"),
            "top": _write(empty, tables / "MLStockLab_top.csv"),
            "bottom": _write(empty, tables / "MLStockLab_bottom.csv"),
            "quintiles": _write(empty, tables / "MLStockLab_quintile_returns.csv"),
            "quintile_metrics": _write(empty, tables / "MLStockLab_quintile_metrics.csv"),
            "prediction_metrics": _write(prediction_metrics, tables / "MLStockLab_prediction_metrics.csv"),
            "long_short_diagnostics": _write(empty, tables / "MLStockLab_long_short_diagnostics.csv"),
            "status": _write(panel_status, tables / "MLStockLab_status.csv"),
            "metrics": _write(metrics, tables / "MLStockLab_metrics.csv"),
        }
        return {"status": "NO_SIGNALS", "paths": paths, "panel": panel, "signals": empty, "metrics": metrics, "status_report": panel_status}

    fitted_index = X.index
    try:
        fair_value = PeerImpliedValuator(model=model).fit_predict(X, y)
    except Exception as exc:
        empty = pd.DataFrame()
        reason = f"SIGNAL_GENERATION_FAILED:{type(exc).__name__}"
        status_values = panel_status.iloc[0].to_dict()
        metrics = pd.DataFrame([{
            "status": "NO_SIGNALS",
            "reason": reason,
            "model": model,
            "rows": 0,
            "features": _csv_value(features),
            "feature_count": len(features),
            "feature_blocks": _csv_value(feature_blocks),
            "universe": universe or "",
            "tickers": _csv_value(tickers),
            "target": "market_value",
            "target_selected": target,
            "target_source": proxy_source,
            "panel_rows": status_values.get("panel_rows"),
            "ticker_count": status_values.get("ticker_count"),
            "date_count": status_values.get("date_count"),
            "min_date": status_values.get("min_date"),
            "max_date": status_values.get("max_date"),
            "r2_os": pd.NA,
            "sharpe_long_short": pd.NA,
            "volatility_long_short": pd.NA,
            "avg_names_long_short": pd.NA,
        }])
        prediction_metrics = pd.DataFrame([{
            "status": "SKIPPED",
            "reason": reason,
            "expected_return_model": "rf",
            "r2_os": pd.NA,
            "test_rows": 0,
        }])
        paths = {
            "panel": _write(panel, tables / "MLStockLab_panel.csv"),
            "signals": _write(empty, tables / "MLStockLab_signals.csv"),
            "top": _write(empty, tables / "MLStockLab_top.csv"),
            "bottom": _write(empty, tables / "MLStockLab_bottom.csv"),
            "quintiles": _write(empty, tables / "MLStockLab_quintile_returns.csv"),
            "quintile_metrics": _write(empty, tables / "MLStockLab_quintile_metrics.csv"),
            "prediction_metrics": _write(prediction_metrics, tables / "MLStockLab_prediction_metrics.csv"),
            "long_short_diagnostics": _write(empty, tables / "MLStockLab_long_short_diagnostics.csv"),
            "status": _write(panel_status, tables / "MLStockLab_status.csv"),
            "metrics": _write(metrics, tables / "MLStockLab_metrics.csv"),
        }
        return {"status": "NO_SIGNALS", "paths": paths, "panel": panel, "signals": empty, "metrics": metrics, "status_report": panel_status}
    signals = panel.loc[fitted_index].copy()
    signals["fair_value_hat"] = fair_value
    signals["mispricing_rel"] = compute_relative_mispricing(signals["fair_value_hat"], signals["market_value"])
    signals["zscore"] = cross_sectional_zscore(signals["mispricing_rel"], signals["date"] if "date" in signals.columns else None)
    signals = rank_scores(signals, "zscore", ascending=False)
    signals["signal"] = signals["zscore"]
    signals["score"] = signals["zscore"]
    top, bottom = top_bottom(signals, "zscore", n=25)
    qret = make_quantile_portfolios(signals, "zscore", return_col="forward_return" if "forward_return" in signals.columns else "mispricing_rel", q=5)
    qmetrics = evaluate_quintile_backtest(qret)

    train_idx, test_idx = temporal_train_test_split(signals)
    split_meta = describe_temporal_split(signals, train_idx, test_idx)
    prediction_row: dict[str, Any] = {
        **split_meta,
        "status": "SKIPPED",
        "reason": "",
        "expected_return_model": "rf",
        "r2_os": pd.NA,
    }
    if "forward_return" in signals.columns and len(train_idx) > 5 and len(test_idx) > 0:
        pred_features = [c for c in features if c in signals.columns]
        yret = pd.to_numeric(signals["forward_return"], errors="coerce")
        if pred_features and yret.loc[train_idx].notna().sum() > 5 and yret.loc[test_idx].notna().sum() > 0:
            try:
                pred_model = ExpectedReturnModel(model="rf").fit(signals.loc[train_idx, pred_features], yret.loc[train_idx])
                preds = pred_model.predict(signals.loc[test_idx, pred_features])
                prediction_row["status"] = "OK"
                prediction_row["r2_os"] = oos_r2(yret.loc[test_idx], preds)
            except Exception as exc:
                prediction_row["reason"] = f"PREDICTION_FAILED:{type(exc).__name__}"
        else:
            prediction_row["reason"] = "INSUFFICIENT_RETURN_SPLIT"
    else:
        prediction_row["reason"] = "FORWARD_RETURN_UNAVAILABLE_OR_SPLIT_TOO_SMALL"
    prediction_metrics = pd.DataFrame([prediction_row])

    if qret.empty:
        ls = pd.Series(dtype=float)
        ls_diag = pd.DataFrame([{
            "status": "NO_QUINTILE_RETURNS",
            "reason": "Quantile return artifact is empty",
            "long_short_mean_return": pd.NA,
            "long_short_volatility": pd.NA,
            "long_short_sharpe": pd.NA,
            "avg_names_long_short": pd.NA,
        }])
    else:
        ls = qret[qret["quantile"].astype(str).eq("long_short")]["return"] if "quantile" in qret else pd.Series(dtype=float)
        ls_names = qret[qret["quantile"].astype(str).eq("long_short")]["name_count"] if "name_count" in qret else pd.Series(dtype=float)
        ls_diag = pd.DataFrame([{
            "status": "OK" if not ls.empty else "NO_LONG_SHORT_LEG",
            "reason": "OK" if not ls.empty else "Missing Q5-Q1 rows",
            "long_short_mean_return": pd.to_numeric(ls, errors="coerce").mean() if not ls.empty else pd.NA,
            "long_short_volatility": pd.to_numeric(ls, errors="coerce").std(ddof=0) if not ls.empty else pd.NA,
            "long_short_sharpe": sharpe_ratio(ls, periods_per_year=12 if len(ls) < 80 else 252),
            "avg_names_long_short": pd.to_numeric(ls_names, errors="coerce").mean() if not ls_names.empty else pd.NA,
        }])

    run_status = "OK" if not signals.empty and not qret.empty else ("NO_SIGNALS" if signals.empty else "NO_QUINTILE_RETURNS")
    if run_status == "OK" and str(ls_diag["status"].iloc[0]) != "OK":
        run_status = str(ls_diag["status"].iloc[0])
    run_reason = "OK" if run_status == "OK" else str(ls_diag["reason"].iloc[0])
    status_values = panel_status.iloc[0].to_dict()
    metrics = pd.DataFrame([{
        "status": run_status,
        "reason": run_reason,
        "model": model,
        "rows": len(signals),
        "features": ",".join(features),
        "feature_count": len(features),
        "feature_blocks": _csv_value(feature_blocks),
        "universe": universe or "",
        "tickers": _csv_value(tickers),
        "target": "market_value",
        "target_selected": target,
        "target_source": proxy_source,
        "panel_rows": status_values.get("panel_rows"),
        "ticker_count": status_values.get("ticker_count"),
        "date_count": status_values.get("date_count"),
        "min_date": status_values.get("min_date"),
        "max_date": status_values.get("max_date"),
        "r2_os": prediction_row["r2_os"],
        "sharpe_long_short": ls_diag["long_short_sharpe"].iloc[0],
        "volatility_long_short": ls_diag["long_short_volatility"].iloc[0],
        "avg_names_long_short": ls_diag["avg_names_long_short"].iloc[0],
    }])

    paths = {
        "panel": _write(panel, tables / "MLStockLab_panel.csv"),
        "signals": _write(signals, tables / "MLStockLab_signals.csv"),
        "top": _write(top, tables / "MLStockLab_top.csv"),
        "bottom": _write(bottom, tables / "MLStockLab_bottom.csv"),
        "quintiles": _write(qret, tables / "MLStockLab_quintile_returns.csv"),
        "quintile_metrics": _write(qmetrics, tables / "MLStockLab_quintile_metrics.csv"),
        "prediction_metrics": _write(prediction_metrics, tables / "MLStockLab_prediction_metrics.csv"),
        "long_short_diagnostics": _write(ls_diag, tables / "MLStockLab_long_short_diagnostics.csv"),
        "status": _write(panel_status, tables / "MLStockLab_status.csv"),
        "metrics": _write(metrics, tables / "MLStockLab_metrics.csv"),
    }
    return {
        "status": run_status,
        "paths": paths,
        "panel": panel,
        "signals": signals,
        "quintiles": qret,
        "metrics": metrics,
        "prediction_metrics": prediction_metrics,
        "status_report": panel_status,
    }
