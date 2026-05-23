"""Smoke checks for the strategic Data Center enhancement."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for candidate in [PROJECT_ROOT, PROJECT_ROOT / "src"]:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from research_platform_core.data_center_catalog import build_target_catalog, summarize_target_catalog
from research_platform_core.data_platform import resolve_data_platform_roots
from research_platform_core.loaders.fama_french import FamaFrenchLoader, parse_fama_french_csv


def main() -> int:
    roots = resolve_data_platform_roots(repo_output_root=PROJECT_ROOT / "output")
    catalog = build_target_catalog(roots.financial_db)
    summary = summarize_target_catalog(catalog)
    assert len(catalog) >= 50
    assert {"equity_universe", "factor_data", "fx", "commodities", "risk_factors"}.issubset(set(catalog["domain"]))
    sample = "skip\n,Mkt-RF,SMB,HML,RF\n202001,1,2,3,0\n\n"
    parsed = parse_fama_french_csv(sample, "monthly")
    assert len(parsed) == 1
    ff_catalog = FamaFrenchLoader(roots.financial_db, roots.repo_output).catalog()
    assert "FF3_daily" in set(ff_catalog["dataset"])
    print(summary.to_string(index=False))
    print("data_center_enhancement_smoke_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
