from __future__ import annotations

from pathlib import Path
import tempfile
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for candidate in [PROJECT_ROOT, PROJECT_ROOT / "src"]:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from research_platform_core.data_center_catalog import build_target_catalog
from research_platform_core.loaders.bditalia_client import BancaDItaliaClient, normalize_bditalia_frame
from research_platform_core.loaders.ecb_client import EcbClient, normalize_ecb_frame
from research_platform_core.macro_features import add_standard_transformations, build_inflation_nowcasting_dataset


def test_ecb_url_and_normalization(tmp_path: Path) -> None:
    client = EcbClient(financial_db_root=tmp_path, output_root=tmp_path)
    assert client.build_url("FM", "D.U2.EUR.4F.KR.DFR.LEV").endswith("/data/FM/D.U2.EUR.4F.KR.DFR.LEV")
    frame = pd.DataFrame({"KEY": ["FM.D.U2.EUR.4F.KR.DFR.LEV"], "TIME_PERIOD": ["2024-01-01"], "OBS_VALUE": ["4.0"]})
    out = normalize_ecb_frame(frame, "FM", "D.U2.EUR.4F.KR.DFR.LEV", "https://example.test")
    assert out["date"].notna().all()
    assert float(out["value"].iloc[0]) == 4.0


def test_bditalia_url_and_long_parser(tmp_path: Path) -> None:
    client = BancaDItaliaClient(financial_db_root=tmp_path, output_root=tmp_path)
    assert client.build_url("AGGM0500").endswith("/IT/CSV/DATA/CUBE/BANKITALIA/DIFF/AGGM0500")
    frame = pd.DataFrame({"DATA": ["2024-01-31"], "PRESTITI": ["1,2"], "DEPOSITI": ["2,3"]})
    out = normalize_bditalia_frame(frame, "AGGM0500", "sample", "https://example.test")
    assert len(out) == 2
    assert set(out["item_code"]) == {"prestiti", "depositi"}


def test_macro_feature_builder_with_lineage() -> None:
    ecb = pd.DataFrame({"date": pd.date_range("2023-01-31", periods=14, freq="ME"), "dataset": "hicp", "value": range(14)})
    bdi = pd.DataFrame({"date": pd.date_range("2023-01-31", periods=14, freq="ME"), "dataset": "credit", "value": range(10, 24)})
    panel = build_inflation_nowcasting_dataset(ecb_hicp=ecb, bdi_credit=bdi, bdi_deposits=bdi, add_transforms=False)
    assert not panel.empty
    transformed = add_standard_transformations(panel, yoy=True, log_diff=False)
    assert "feature_metadata" in transformed.attrs
    assert any(col.endswith("_yoy") for col in transformed.columns)


def test_catalog_includes_official_macro(tmp_path: Path) -> None:
    catalog = build_target_catalog(tmp_path)
    assert "official_macro" in set(catalog["domain"])


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        test_ecb_url_and_normalization(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_bditalia_url_and_long_parser(Path(tmp))
    test_macro_feature_builder_with_lineage()
    with tempfile.TemporaryDirectory() as tmp:
        test_catalog_includes_official_macro(Path(tmp))
    print("official_macro_tests_OK")
