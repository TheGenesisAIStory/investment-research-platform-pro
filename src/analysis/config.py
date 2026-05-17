"""Shared configuration for Analysis Studio."""

from __future__ import annotations

import os
from pathlib import Path

DB_BASE = Path(
    os.environ.get(
        "ML_TRADING_DB_BASE",
        "/Users/itsgennymac/Library/CloudStorage/"
        "GoogleDrive-sfn.gns@gmail.com/Il mio Drive/Database Finanziario",
    )
).expanduser()

os.environ.setdefault("DATA_PATH", str(DB_BASE))

ANALYSIS_OUTPUTS = DB_BASE / "analysis_outputs"
NOTEBOOK_EXPORTS = DB_BASE / "notebook_exports"
CSV_EXPORTS = NOTEBOOK_EXPORTS / "csv"
HTML_EXPORTS = NOTEBOOK_EXPORTS / "html"
MARKDOWN_EXPORTS = NOTEBOOK_EXPORTS / "markdown"
CHART_EXPORTS = NOTEBOOK_EXPORTS / "charts"
LOG_EXPORTS = DB_BASE / "logs"

ANALYSIS_REGISTRY = {
    "stock_screening": {
        "title": "Stock Screening",
        "output_dir": ANALYSIS_OUTPUTS / "stock_screening",
    },
    "portfolio_risk": {
        "title": "Portfolio Risk",
        "output_dir": ANALYSIS_OUTPUTS / "portfolio_risk",
    },
    "earnings_analysis": {
        "title": "Earnings Analysis",
        "output_dir": ANALYSIS_OUTPUTS / "earnings_analysis",
    },
    "portfolio_builder": {
        "title": "Portfolio Builder",
        "output_dir": ANALYSIS_OUTPUTS / "portfolio_builder",
    },
    "technical_analysis": {
        "title": "Technical Analysis",
        "output_dir": ANALYSIS_OUTPUTS / "technical_analysis",
    },
    "competitive_analysis": {
        "title": "Competitive Analysis",
        "output_dir": ANALYSIS_OUTPUTS / "competitive_analysis",
    },
    "quantitative_research": {
        "title": "Quantitative Research",
        "output_dir": ANALYSIS_OUTPUTS / "quantitative_research",
    },
    "macro_analysis": {
        "title": "Macro Analysis",
        "output_dir": ANALYSIS_OUTPUTS / "macro_analysis",
    },
}

UNIVERSES = {
    "core": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "JPM", "XOM", "UNH", "SPY"],
    "sp500": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "BRK-B", "LLY", "JPM", "AVGO"],
    "technology": ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AVGO", "AMD", "CRM", "ORCL", "ADBE"],
    "semiconductors": ["NVDA", "AMD", "AVGO", "INTC", "QCOM", "TXN", "MU", "ASML", "TSM", "ARM"],
    "banks": ["JPM", "BAC", "WFC", "C", "GS", "MS", "USB", "PNC", "BK", "SCHW"],
    "macro": ["SPY", "QQQ", "IWM", "TLT", "GLD", "UUP", "USO", "EFA", "EEM", "HYG"],
}

SECTOR_UNIVERSES = {
    "technology": UNIVERSES["technology"],
    "semiconductors": UNIVERSES["semiconductors"],
    "financials": UNIVERSES["banks"],
    "banks": UNIVERSES["banks"],
}


def ensure_base_directories() -> None:
    """Create persistent Analysis Studio folders under DB_BASE."""
    folders = [
        DB_BASE,
        ANALYSIS_OUTPUTS,
        NOTEBOOK_EXPORTS,
        CSV_EXPORTS,
        HTML_EXPORTS,
        MARKDOWN_EXPORTS,
        CHART_EXPORTS,
        LOG_EXPORTS,
        *[meta["output_dir"] for meta in ANALYSIS_REGISTRY.values()],
    ]
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)


def analysis_output_dir(name: str) -> Path:
    """Return and create the output directory for an analysis slug."""
    output_dir = ANALYSIS_REGISTRY[name]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

