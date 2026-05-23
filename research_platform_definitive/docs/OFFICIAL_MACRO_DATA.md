# Official Macro Data Integration

This layer adds reusable official-source clients for macro, monetary and credit-risk research.

## Sources

- ECB Data Portal datasets: https://data.ecb.europa.eu/data/datasets
- ECB SDMX 2.1 REST data API: https://data.ecb.europa.eu/help/api/data
- ECB content negotiation: https://data.ecb.europa.eu/help/api/content-negotiation
- Banca d'Italia Infostat inquiry: https://infostat.bancaditalia.it/inquiry/
- Banca d'Italia BDS overview: https://www.bancaditalia.it/statistiche/basi-dati/bds/
- Banca d'Italia release calendar: https://www.bancaditalia.it/statistiche/calendario-pubblicazioni/calendario-pubblicazioni.html
- Banca d'Italia QEF 721 on big data, analytics and AI:
  https://www.bancaditalia.it/pubblicazioni/qef/2022-0721/QEF_721_EN.pdf?language_id=1
- Banca d'Italia QEF 693 on big data:
  https://www.bancaditalia.it/pubblicazioni/qef/2022-0693/QEF_693_22.pdf

## Modules

- `research_platform_core.loaders.ecb_client.EcbClient`
  - `get_series(flow_ref, key, start=None, end=None, format="csv", **params)`
  - Helpers for euro area HICP, ECB policy rates and MFI balance sheet monetary aggregates.
- `research_platform_core.loaders.bditalia_client.BancaDItaliaClient`
  - BDS/Infostat A2A ZIP/CSV downloads for cubes and publications.
  - Helpers for credit, deposits and public debt/public finance datasets.
- `research_platform_core.macro_features`
  - `build_inflation_nowcasting_dataset(...)`
  - `build_credit_risk_macro_dataset(...)`

## Sync

Dry run:

```bash
python scripts/sync_official_macro.py
```

Download official macro presets and build feature panels:

```bash
python scripts/sync_official_macro.py --execute --build-features --start 2010-01
```

Outputs are written under:

```text
Database Finanziario/
  OfficialMacro/
    ECB/
    BancaItalia/
    features/
```

Each persisted dataset has a `.metadata.json` sidecar with source, URL, row count and write timestamp. Keep these sidecars with model artifacts for audit/regulatory reproducibility.

## Notes For ML And Regulatory Use

- ECB series keep `flow_ref`, `series_key`, `source_url` and `retrieved_at`.
- Banca d'Italia series keep `publication`, `source_file`, `source_url` and `retrieved_at`.
- Feature panels align to month-end by default, forward-fill short release lags and store transformation lineage in `DataFrame.attrs["feature_metadata"]`.
- Before freezing a model snapshot, check the Banca d'Italia release calendar and the ECB dataset metadata/revision notes.
