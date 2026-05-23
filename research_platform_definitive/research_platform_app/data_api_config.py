from __future__ import annotations

from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "data_api_control.yaml"

DEFAULT_CONFIG: dict[str, Any] = {
    "app": {
        "title": "Data/API Control Center",
        "default_inventory_limit": 5000,
        "health_warning_threshold": 0.75,
        "health_success_threshold": 0.9,
    },
    "theme": {
        "primary_color": "#1E3A8A",
        "success_color": "#10B981",
        "warning_color": "#F59E0B",
        "error_color": "#EF4444",
        "neutral_color": "#6B7280",
    },
    "batch_download": {
        "default_format": "csv",
        "max_preview_rows": 2000,
        "max_interactive_export_files": 250,
        "output_subdir": "batch_downloads",
        "retry_failed": True,
    },
    "market_context": {
        "default_currency": "USD",
        "currencies": {
            "USD": {"benchmark": "SPY", "fx_pair": "DX-Y.NYB", "overlays": ["UUP", "TLT", "GLD", "DBC"]},
            "EUR": {"benchmark": "SX5E.DE", "fx_pair": "EURUSD=X", "overlays": ["EURUSD=X", "FEZ", "EWI", "GLD"]},
        },
    },
    "schedules": {
        "api_health_minutes": 15,
        "market_data_sync": "daily",
        "fundamentals_refresh": "weekly",
        "full_drive_backup": "monthly",
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_data_api_config(path: Path | str | None = None) -> dict[str, Any]:
    config_path = Path(path or CONFIG_PATH).expanduser()
    if not config_path.exists():
        return DEFAULT_CONFIG
    try:
        import yaml
    except Exception:
        return DEFAULT_CONFIG
    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception:
        loaded = {}
    return _deep_merge(DEFAULT_CONFIG, loaded)
