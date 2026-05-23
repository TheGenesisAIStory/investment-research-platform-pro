"""Sync official ECB and Banca d'Italia macro datasets.

Default mode is a dry run that writes the target catalog only. Use --execute to
perform network downloads.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for candidate in [PROJECT_ROOT, PROJECT_ROOT / "src"]:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from research_platform_core.data_center_catalog import build_target_catalog, summarize_target_catalog
from research_platform_core.data_platform import resolve_data_platform_roots
from research_platform_core.loaders.bditalia_client import BancaDItaliaClient
from research_platform_core.loaders.ecb_client import EcbClient
from research_platform_core.macro_features import (
    build_credit_risk_macro_dataset,
    build_inflation_nowcasting_dataset,
    save_macro_feature_panel,
)


def _split(value: str) -> list[str]:
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", default="ecb,bditalia", help="Comma-separated: ecb,bditalia")
    parser.add_argument("--ecb-presets", default="hicp_euro_area_yoy,policy_rate_mro,policy_rate_deposit,m2_notional_stock_index,m3_notional_stock_index")
    parser.add_argument("--bditalia-presets", default="credit_growth,deposit_volumes,public_debt,official_rates_bdi")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--format", default="csv", choices=["csv", "parquet"])
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--build-features", action="store_true", help="Build combined ML panels after syncing.")
    args = parser.parse_args()

    roots = resolve_data_platform_roots(repo_output_root=PROJECT_ROOT / "output")
    catalog = build_target_catalog(roots.financial_db)
    official_catalog = catalog[catalog["domain"].eq("official_macro")]
    out_dir = roots.repo_output / "data_quality"
    out_dir.mkdir(parents=True, exist_ok=True)
    official_catalog.to_csv(out_dir / "OfficialMacro_target_catalog.csv", index=False)
    summarize_target_catalog(catalog).to_csv(out_dir / "DataCenter_target_summary.csv", index=False)
    print(f"Official macro catalog written: {out_dir / 'OfficialMacro_target_catalog.csv'}")

    if not args.execute:
        print("Dry run only. Re-run with --execute to download official macro data.")
        return 0

    selected = {source.lower() for source in _split(args.sources)}
    ecb_frames: dict[str, object] = {}
    bdi_frames: dict[str, object] = {}
    if "ecb" in selected:
        ecb = EcbClient(roots.financial_db, roots.repo_output)
        manifest = ecb.sync_presets(_split(args.ecb_presets), start=args.start, end=args.end, refresh=args.refresh, fmt=args.format)
        print(manifest.to_string(index=False))
        for name in _split(args.ecb_presets):
            path = roots.financial_db / "OfficialMacro" / "ECB" / f"{name}.{args.format}"
            if path.exists() and path.suffix == ".csv":
                import pandas as pd

                ecb_frames[name] = pd.read_csv(path)
    if "bditalia" in selected:
        bdi = BancaDItaliaClient(roots.financial_db, roots.repo_output)
        manifest = bdi.sync_presets(_split(args.bditalia_presets), start=args.start, end=args.end, refresh=args.refresh, fmt=args.format)
        print(manifest.to_string(index=False))
        for name in _split(args.bditalia_presets):
            path = roots.financial_db / "OfficialMacro" / "BancaItalia" / f"{name}.{args.format}"
            if path.exists() and path.suffix == ".csv":
                import pandas as pd

                bdi_frames[name] = pd.read_csv(path)

    if args.build_features:
        import pandas as pd

        empty = pd.DataFrame()
        feature_dir = roots.financial_db / "OfficialMacro" / "features"
        inflation = build_inflation_nowcasting_dataset(
            ecb_hicp=ecb_frames.get("hicp_euro_area_yoy", empty if "ecb" not in selected else None),
            bdi_credit=bdi_frames.get("credit_growth", empty if "bditalia" not in selected else None),
            bdi_deposits=bdi_frames.get("deposit_volumes", empty if "bditalia" not in selected else None),
        )
        credit_risk = build_credit_risk_macro_dataset(
            ecb_policy_rates=empty if "ecb" not in selected else None,
            ecb_mfi=empty if "ecb" not in selected else None,
            bdi_credit=bdi_frames.get("credit_growth", empty if "bditalia" not in selected else None),
            bdi_public_debt=bdi_frames.get("public_debt", empty if "bditalia" not in selected else None),
            ecb_client=EcbClient(roots.financial_db, roots.repo_output) if "ecb" in selected else None,
        )
        print(f"Inflation nowcasting rows: {len(inflation)}")
        print(f"Credit risk macro rows: {len(credit_risk)}")
        save_macro_feature_panel(inflation, feature_dir / "inflation_nowcasting_panel.csv", fmt="csv")
        save_macro_feature_panel(credit_risk, feature_dir / "credit_risk_macro_panel.csv", fmt="csv")

    print("Official macro sync complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
