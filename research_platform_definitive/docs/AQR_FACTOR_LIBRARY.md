# AQR Factor Library Integration

The project integrates the public AQR Data Library as a Drive-first factor provider.

## Source

Official public index pages:

- `https://www.aqr.com/Insights/Datasets`
- `https://www.aqr.com/Insights/Datasets?page=2`

The provider discovers dataset detail pages, extracts Excel links, caches raw workbooks and parses sheets that contain a `DATE` row into date-indexed pandas panels.

## Public API

```python
from src.research_platform_core import AqrFactorProvider, get_aqr_factor_panel, get_all_factors_panel

panel = get_aqr_factor_panel(
    slugs=["quality_minus_junk_factors_monthly", "betting_against_beta_equity_factors_monthly"],
    start="2000-01-01",
)
```

For ML workflows:

```python
from ml_stock_lab import load_aqr_factor_panel

aqr = load_aqr_factor_panel(["quality_minus_junk_factors_monthly"])
```

## Artifacts

The refresh job writes:

- `output/aqr_factors/tables/AQRFactorDiscovery.csv`
- `output/aqr_factors/tables/AQRFactorRefreshLog.csv`
- `output/aqr_factors/tables/AQRFactorPanel.csv`
- `output/aqr_factors/tables/AQRFactorManifest.json`

When `Database Finanziario` is available, raw and processed cache files are stored under:

- `Database Finanziario/alpha_factor_library/aqr/raw`
- `Database Finanziario/alpha_factor_library/aqr/processed`
- `Database Finanziario/alpha_factor_library/aqr/catalog`

## Orchestration

Streamlit/CLI job:

```bash
python research_platform_app/scheduler.py --once --jobs aqr_factor_library_refresh
```

`top_n=0` means refresh all discovered AQR workbooks. For a light smoke run, use `top_n=1`.

## Notes

- AQR workbooks often contain several sheets; only sheets with a detectable date table are parsed.
- Column names are prefixed as `dataset__sheet__column` to avoid collisions across datasets and markets.
- The provider does not fabricate factors. Failed/empty parses are surfaced in `AQRFactorRefreshLog.csv`.
