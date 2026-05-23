"""Smoke checks for the open banking data integration."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from research_platform_core import ITALIAN_LISTED_BANK_TICKERS, normalize_bank_name, run_banks_data_pipeline


def main() -> int:
    assert normalize_bank_name("Intesa Sanpaolo S.p.A.").startswith("INTESA")
    assert "INTESA SANPAOLO" in ITALIAN_LISTED_BANK_TICKERS
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out"
        cache = Path(tmp) / "cache"
        artifacts = run_banks_data_pipeline(
            output_dir=out,
            cache_dir=cache,
            include_market=False,
            include_ecb_macro=False,
        )
        assert "banks_universe" in artifacts
        assert (out / "banks_universe.csv").exists()
        assert (out / "banks_data.sqlite").exists()
    print("banking_data_smoke_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

