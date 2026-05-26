from __future__ import annotations

import pandas as pd

from research_platform_core import construct_it_local_factors, download_ff_factors
import research_platform_core.aqr_factors as aqr_factors


def test_ff_download_us_with_fallback_cache(tmp_path, monkeypatch) -> None:
    def fail_datareader(dataset: str, start_date: str) -> pd.DataFrame:
        raise RuntimeError("force direct fallback")

    def fake_zip(dataset: str) -> pd.DataFrame:
        if "Momentum" in dataset:
            return pd.DataFrame({"date": pd.to_datetime(["2000-01-31", "2000-02-29"]), "Mom": [2.0, 3.0]})
        return pd.DataFrame(
            {
                "date": pd.to_datetime(["1999-12-31", "2000-01-31", "2000-02-29"]),
                "Mkt-RF": [1.0, 4.0, 5.0],
                "SMB": [0.1, 0.2, 0.3],
                "HML": [0.4, 0.5, 0.6],
                "RMW": [0.0, 0.1, 0.2],
                "CMA": [0.0, 0.1, 0.2],
                "RF": [0.01, 0.01, 0.01],
            }
        )

    monkeypatch.setattr(aqr_factors, "_read_french_datareader", fail_datareader)
    monkeypatch.setattr(aqr_factors, "_read_french_zip", fake_zip)

    frame = download_ff_factors("US", "FF6", start_date="2000-01-01", output_root=tmp_path, refresh=True)
    assert {"date", "MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM", "RF"}.issubset(frame.columns)
    assert float(frame["MKT_RF"].iloc[0]) == 0.04
    assert (tmp_path / "ff_factors" / "US_FF5_MOM.parquet").exists()

    cached = download_ff_factors("US", "FF6", start_date="2000-01-01", output_root=tmp_path, refresh=False)
    assert len(cached) == len(frame)


def test_ff_download_europe_ff3(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        aqr_factors,
        "_read_french_datareader",
        lambda dataset, start_date: pd.DataFrame(
            {
                "date": pd.to_datetime(["2000-01-31"]),
                "Mkt-RF": [2.5],
                "SMB": [0.5],
                "HML": [0.2],
                "RF": [0.01],
            }
        ),
    )
    frame = download_ff_factors("EU", "FF3", start_date="2000-01-01", output_root=tmp_path, refresh=True)
    assert list(frame.columns) == ["date", "MKT_RF", "SMB", "HML", "RF"]
    assert float(frame["MKT_RF"].iloc[0]) == 0.025


def test_ff_it_local_construction_from_factor_panel(tmp_path) -> None:
    rows = []
    for month in pd.period_range("2024-01", periods=3, freq="M"):
        for idx in range(8):
            rows.append(
                {
                    "date": month.to_timestamp("M"),
                    "ticker": f"IT{idx}.MI",
                    "country": "IT",
                    "market_cap": 100 + idx,
                    "book_to_market": idx / 10,
                    "momentum_12m_1m": idx / 20,
                    "forward_return_21d": idx / 100,
                }
            )
    panel = pd.DataFrame(rows)
    frame = construct_it_local_factors(panel, output_root=tmp_path, write=True)
    assert {"date", "MKT_RF", "SMB", "HML", "MOM", "RF"}.issubset(frame.columns)
    assert len(frame) == 3
    assert (tmp_path / "ff_factors" / "IT_LOCAL_FACTORS.parquet").exists()
