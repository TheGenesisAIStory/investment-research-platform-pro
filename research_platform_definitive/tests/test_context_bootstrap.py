from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
from streamlit.testing.v1 import AppTest

from research_platform_core.data_platform import shared_database_contract
from research_platform_app.data_bootstrap import data_readiness
from research_platform_app.support import get_platform_context, ohlcv_coverage_counts
from research_platform_app.ui_ops import job_status_board


def test_packages_import_from_clean_cwd(tmp_path: Path) -> None:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import research_platform_core, ml_stock_lab, smart_money_engine; print('imports_ok')",
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "imports_ok" in result.stdout


def test_platform_context_falls_back_to_not_set() -> None:
    context = get_platform_context({"reporting_currency": "EUR", "selected_ticker": "AAPL"})

    assert context["reporting_currency"] == "EUR"
    assert context["selected_ticker"] == "AAPL"
    assert context["benchmark_ticker"] == "SPY"
    assert context["active_screener_name"] == "not set"


def test_data_readiness_reports_missing_without_crashing(tmp_path: Path) -> None:
    roots = {"financial_db": tmp_path / "missing_financial_db", "workspace": tmp_path / "output"}
    readiness = data_readiness(roots)

    assert not readiness.empty
    assert {"dataset", "status", "path", "required_for"}.issubset(readiness.columns)
    assert readiness["status"].isin(["missing", "available"]).all()


def test_shared_database_contract_is_single_root(tmp_path: Path) -> None:
    db = tmp_path / "Database Finanziario"
    output = tmp_path / "output"
    (db / "MarketData" / "OHLCV" / "manifests").mkdir(parents=True)
    (output / "smart_money").mkdir(parents=True)
    contract = shared_database_contract(db, output)

    assert {"domain", "path", "exists", "marker_exists"}.issubset(contract.columns)
    assert contract["path"].astype(str).str.contains(str(db)).any()
    assert contract["path"].astype(str).str.contains(str(output)).any()


def test_ohlcv_coverage_counts_exposes_limited_history() -> None:
    manifest = pd.DataFrame(
        [
            {"ticker": "FBYD", "provider_symbol": "FBYD", "coverage_status": "LIMITED_HISTORY"},
            {"ticker": "AAA", "provider_symbol": "AAA", "coverage_status": "OK"},
            {"ticker": "BBB", "provider_symbol": "BBB", "coverage_status": "NETWORK_TIMEOUT"},
            {"ticker": "CCC", "provider_symbol": "CCC", "coverage_status": "DELISTED"},
        ]
    )
    counts = ohlcv_coverage_counts(manifest)

    assert counts["LIMITED_HISTORY"] == 1
    assert counts["OK"] == 1
    assert counts["NETWORK_TIMEOUT"] == 1
    assert counts["DELISTED"] == 1


def test_context_bar_renders_in_streamlit_app(tmp_path: Path) -> None:
    script = tmp_path / "context_app.py"
    script.write_text(
        "import sys\n"
        "from pathlib import Path\n"
        f"sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r})\n"
        "import streamlit as st\n"
        "from research_platform_app.support import render_context_bar\n"
        "st.session_state['selected_ticker'] = 'AAPL'\n"
        "render_context_bar(show_actions=False)\n",
        encoding="utf-8",
    )

    app = AppTest.from_file(str(script), default_timeout=10)
    app.run()

    assert not app.exception
    assert "AAPL" in app.markdown[1].value


def test_job_status_board_handles_never_run_jobs() -> None:
    class Job:
        label = "Sample Job"
        tags = ["sample"]
        runner_type = "module"
        enabled = True
        output_domain = "workspace"
        expected_artifacts = [object()]
        description = "sample"

    board = job_status_board({"sample_job": Job()}, runs_df=pd.DataFrame())

    assert board.iloc[0]["last_status"] == "NEVER_RUN"
    assert board.iloc[0]["expected_artifacts"] == 1
