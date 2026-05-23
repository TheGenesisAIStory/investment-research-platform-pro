"""Smoke checks for ECB/Banca d'Italia official macro integration."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for candidate in [PROJECT_ROOT, PROJECT_ROOT / "src"]:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from research_platform_core.data_center_catalog import build_target_catalog
from research_platform_core.loaders.bditalia_client import BancaDItaliaClient, normalize_bditalia_frame
from research_platform_core.loaders.ecb_client import EcbClient, normalize_ecb_frame
from research_platform_core.macro_features import build_credit_risk_macro_dataset, build_inflation_nowcasting_dataset


def main() -> int:
    ecb = EcbClient(financial_db_root=PROJECT_ROOT / "output" / "tmp_financial_db", output_root=PROJECT_ROOT / "output")
    bdi = BancaDItaliaClient(financial_db_root=PROJECT_ROOT / "output" / "tmp_financial_db", output_root=PROJECT_ROOT / "output")
    assert ecb.build_url("ICP", "M.U2.N.000000.4.ANR").endswith("/data/ICP/M.U2.N.000000.4.ANR")
    assert "/DATA/CUBE/BANKITALIA/DIFF/AGGM0500" in bdi.build_url("AGGM0500")

    ecb_sample = pd.DataFrame({"KEY": ["ICP.M.U2.N.000000.4.ANR"], "TIME_PERIOD": ["2024-01"], "OBS_VALUE": ["2.8"]})
    ecb_norm = normalize_ecb_frame(ecb_sample, "ICP", "M.U2.N.000000.4.ANR", "https://example.test")
    assert float(ecb_norm["value"].iloc[0]) == 2.8

    bdi_sample = pd.DataFrame({"DATA": ["2024-01-31", "2024-02-29"], "PRESTITI": ["1,2", "1,3"], "DEPOSITI": ["2,1", "2,2"]})
    bdi_norm = normalize_bditalia_frame(bdi_sample, "AGGM0500", "sample", "https://example.test", category="credit")
    assert {"date", "item_code", "value"}.issubset(bdi_norm.columns)
    assert len(bdi_norm) == 4

    inflation = build_inflation_nowcasting_dataset(ecb_hicp=ecb_norm, bdi_credit=bdi_norm, bdi_deposits=bdi_norm, add_transforms=False)
    credit = build_credit_risk_macro_dataset(ecb_policy_rates=ecb_norm, ecb_mfi=ecb_norm, bdi_credit=bdi_norm, bdi_public_debt=bdi_norm, add_transforms=False)
    assert not inflation.empty
    assert not credit.empty
    catalog = build_target_catalog(PROJECT_ROOT / "output" / "tmp_financial_db")
    assert "official_macro" in set(catalog["domain"])
    print("official_macro_smoke_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
