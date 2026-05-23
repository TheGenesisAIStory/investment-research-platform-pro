#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
streamlit run research_platform_app/app.py

