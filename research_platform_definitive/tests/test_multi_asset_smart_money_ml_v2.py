from __future__ import annotations

import pandas as pd

from research_platform_core.factor_portfolio_baselines import compute_factor_portfolio_baselines
from research_platform_core.model_monitoring import build_model_monitoring_artifacts
from research_platform_core.multi_asset_universe import ingest_multi_asset_universe
from research_platform_core.smart_money import fetch_etf_flows_proxy, fetch_options_put_call_ratio


def test_ingest_multi_asset_universe_writes_manifest_and_snapshot(tmp_path):
    dates = pd.date_range("2024-01-01", periods=70, freq="B")

    def fake_download(symbol: str, start: str, end: str | None) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Open": range(70),
                "High": range(1, 71),
                "Low": range(70),
                "Close": [100 + idx for idx in range(70)],
                "Volume": [1000] * 70,
            },
            index=dates,
        )

    bundle = ingest_multi_asset_universe(
        ["SPY", "EURUSD"],
        start_date="2024-01-01",
        output_dir=tmp_path,
        download_fn=fake_download,
        refresh=True,
    )

    assert len(bundle["manifest"]) == 2
    assert bundle["manifest"]["data_status"].eq("OK").all()
    assert {"pct_1d", "pct_5d", "pct_21d", "pct_63d"}.issubset(bundle["snapshot"].columns)
    assert (tmp_path / "multi_asset_universe" / "MultiAssetUniverseManifest.csv").exists()
    assert (tmp_path / "multi_asset_universe" / "ohlcv" / "SPY.parquet").exists()


def test_smart_money_etf_flows_and_options_proxy(tmp_path):
    dates = pd.date_range("2024-01-01", periods=80, freq="B")
    history = pd.DataFrame({"Close": [100 + idx * 0.5 for idx in range(80)]}, index=dates)
    flows = fetch_etf_flows_proxy(
        tmp_path,
        fetch=True,
        history_frames={"SPY": history},
        ticker_info={"SPY": {"totalAssets": 10_000_000, "sharesOutstanding": 100_000}},
    )
    assert flows["snapshot"].iloc[0]["data_status"] == "OK"
    assert "flow_1m_proxy" in flows["snapshot"].columns
    assert (tmp_path / "smart_money" / "etf_flows" / "etf_flows_snapshot.csv").exists()

    calls = pd.DataFrame({"openInterest": [100, 200]})
    puts = pd.DataFrame({"openInterest": [150, 300]})
    pcr = fetch_options_put_call_ratio(tmp_path, fetch=True, option_frames={"SPY": {"calls": calls, "puts": puts}})
    assert float(pcr.iloc[0]["put_call_ratio_oi"]) == 1.5
    assert (tmp_path / "smart_money" / "options" / "pcr_snapshot.csv").exists()


def test_factor_portfolio_baselines_compute_monthly_controls():
    rows = []
    for month in pd.period_range("2024-01", periods=4, freq="M"):
        for idx in range(50):
            rows.append(
                {
                    "date": month.to_timestamp() + pd.Timedelta(days=idx % 20),
                    "ticker": f"T{idx:03d}",
                    "value_score": idx,
                    "quality_score": 50 - idx,
                    "momentum_score": idx % 25,
                    "forward_return_21d": idx / 1000,
                    "forward_return_63d": idx / 900,
                    "forward_return_252d": idx / 800,
                }
            )
    panel = pd.DataFrame(rows)
    bundle = compute_factor_portfolio_baselines(panel=panel, write=False)
    metrics = bundle["metrics"]
    assert {"long_only", "long_short"}.issubset(set(metrics["strategy"]))
    assert {21, 63, 252}.issubset(set(metrics["horizon"]))
    assert metrics["turnover"].notna().any()


def test_model_monitoring_builds_rolling_ic_artifacts(tmp_path):
    rows = []
    for date in pd.date_range("2024-01-01", periods=8, freq="B"):
        for idx in range(12):
            for model in ["ols", "gbrt"]:
                rows.append(
                    {
                        "date": date,
                        "ticker": f"T{idx:03d}",
                        "model": model,
                        "forward_return": idx / 100,
                        "expected_return": idx / 100 + (0.001 if model == "gbrt" else 0),
                    }
                )
    predictions = pd.DataFrame(rows)
    bundle = build_model_monitoring_artifacts(tmp_path, predictions=predictions, window=3, write=True)
    assert set(bundle["summary"]["model"]) == {"ols", "gbrt"}
    assert "rank_ic_rolling_12m" in bundle["rolling"].columns
    assert (tmp_path / "ml_lab" / "model_monitoring" / "model_monitoring_summary.csv").exists()

