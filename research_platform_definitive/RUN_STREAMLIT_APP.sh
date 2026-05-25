#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python -c "import research_platform_core, ml_stock_lab, smart_money_engine" >/dev/null 2>&1 || python -m pip install -e ".[dev]"
streamlit run research_platform_app/app.py
