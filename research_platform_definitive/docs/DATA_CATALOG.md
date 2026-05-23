# Data Catalog

Strategic Data Center target coverage for quantitative ML research.

## Domains

| Domain | Target Storage | Refresh |
|---|---|---|
| Fama-French factors | `Database Finanziario/Factors/FamaFrench/` | monthly |
| AQR factors | `Database Finanziario/alpha_factor_library/aqr/` | monthly |
| Risk factors | `Database Finanziario/RiskFactors/` | daily/monthly by source |
| Equity universes | `Database Finanziario/Equities/<region>/<universe>/` | daily prices, weekly fundamentals |
| FX majors | `Database Finanziario/FX/` | daily |
| Commodities | `Database Finanziario/Commodities/` | daily |

## Implemented Loaders

```text
src/research_platform_core/loaders/fama_french.py
src/research_platform_core/loaders/aqr_data.py
src/research_platform_core/loaders/equity_universe.py
src/research_platform_core/loaders/fx_commodities.py
src/research_platform_core/loaders/risk_factors.py
```

## Coverage Reports

```bash
python scripts/validate_data.py
```

Writes:

```text
output/data_quality/DataCenter_target_catalog.csv
output/data_quality/DataCenter_target_summary.csv
output/data_quality/DataCenter_stale_inventory.csv
```

## Initial Population

Dry run:

```bash
python scripts/initial_setup.py
```

Smoke population:

```bash
python scripts/initial_setup.py --execute --max-items 5
```

Full population is intentionally explicit:

```bash
python scripts/initial_setup.py \
  --equities sp500,eurostoxx50,ftsemib,nasdaq100,ftse100,dax40 \
  --factors ff,aqr,risk \
  --fx majors \
  --commodities all \
  --max-items 0 \
  --execute
```

Expect a long run and significant storage usage.
