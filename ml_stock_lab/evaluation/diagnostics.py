"""Operational diagnostics and status files for ML stock experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional
import json

import pandas as pd

from .performance import quintile_risk_report


@dataclass
class PanelValidationResult:
    """Coverage diagnostics for a date/ticker stock panel."""

    ok: bool
    min_date: Optional[str]
    max_date: Optional[str]
    n_dates: int
    n_tickers: int
    reason: Optional[str] = None
    extra: dict[str, Any] | None = None


@dataclass
class ExperimentStatus:
    """Serializable status for a notebook or pipeline run."""

    status: str
    reason: Optional[str] = None
    details: dict[str, Any] | None = None


def _series_from_column_or_index(panel: pd.DataFrame, name: str) -> pd.Series:
    if name in panel.columns:
        return panel[name]
    if isinstance(panel.index, pd.MultiIndex) and name in panel.index.names:
        return pd.Series(panel.index.get_level_values(name), index=panel.index)
    if panel.index.name == name:
        return pd.Series(panel.index, index=panel.index)
    return pd.Series(dtype=object)


def validate_panel(
    panel: pd.DataFrame,
    min_dates: int = 24,
    min_tickers: int = 20,
    date_col: str = "date",
    ticker_col: str = "ticker",
) -> PanelValidationResult:
    """Check temporal coverage and universe width for a stock panel."""
    if panel is None or panel.empty:
        return PanelValidationResult(False, None, None, 0, 0, "panel_empty", {"min_dates": min_dates, "min_tickers": min_tickers})

    dates = pd.to_datetime(_series_from_column_or_index(panel, date_col), errors="coerce").dropna()
    tickers = _series_from_column_or_index(panel, ticker_col).dropna().astype(str)
    n_dates = int(dates.nunique())
    n_tickers = int(tickers[tickers.ne("")].nunique())
    min_date = dates.min().date().isoformat() if not dates.empty else None
    max_date = dates.max().date().isoformat() if not dates.empty else None

    reasons = []
    if n_dates < min_dates:
        reasons.append("too_few_dates")
    if n_tickers < min_tickers:
        reasons.append("too_few_tickers")

    return PanelValidationResult(
        ok=not reasons,
        min_date=min_date,
        max_date=max_date,
        n_dates=n_dates,
        n_tickers=n_tickers,
        reason=";".join(reasons) if reasons else None,
        extra={"min_dates": min_dates, "min_tickers": min_tickers},
    )


def should_run_experiment(panel: pd.DataFrame, **kwargs: Any) -> bool:
    """Return whether an experiment should run based on panel validation."""
    return validate_panel(panel, **kwargs).ok


def build_status_from_signals(
    signals: Optional[pd.DataFrame],
    quintile_returns: Optional[pd.DataFrame],
) -> ExperimentStatus:
    """Build experiment status from signal and quintile artifacts."""
    if signals is None or signals.empty:
        return ExperimentStatus(status="error", reason="signals_empty", details={"signal_rows": 0})
    if quintile_returns is None or quintile_returns.empty:
        return ExperimentStatus(
            status="error",
            reason="quintile_returns_empty",
            details={"signal_rows": len(signals), "quintile_return_rows": 0},
        )
    return ExperimentStatus(
        status="ok",
        details={"signal_rows": len(signals), "quintile_return_rows": len(quintile_returns)},
    )


def save_status_file(status: ExperimentStatus, output_root: Path) -> Path:
    """Save `MLStockLab_status.json` under `output_root` and return its path."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "MLStockLab_status.json"
    path.write_text(json.dumps(asdict(status), indent=2, default=str), encoding="utf-8")
    return path


__all__ = [
    "ExperimentStatus",
    "PanelValidationResult",
    "build_status_from_signals",
    "quintile_risk_report",
    "save_status_file",
    "should_run_experiment",
    "validate_panel",
]
