# API Usage

The Data Center follows the same operating policy as the app:

1. Drive-first.
2. Local cache-second.
3. Provider/API-last.
4. Incremental refresh before full refresh.
5. Write metadata and manifests after every run.

## Provider Orchestration

```text
src/research_platform_core/api_orchestrator.py
src/research_platform_core/cache_manager.py
src/research_platform_core/batch_downloader.py
config/rate_limits.yaml
```

`APIOrchestrator` tries providers by priority and logs success/failure in JSONL.
`DataCache` stores results with TTL metadata.
`BatchDownloader` runs bounded parallel jobs with requests-per-minute throttling.

## Provider Defaults

| Provider | Default Use | Cost Tier |
|---|---|---|
| yfinance | prices, FX, commodities, basic fundamentals | free |
| FRED | rates, credit, inflation proxies | free |
| Ken French Data Library | academic factors | free |
| AQR Data Library | academic style factors | free |
| Alpha Vantage/FMP/Polygon | fallback/enrichment | key-dependent |

## Secret Safety

Credential values are not exported to CSV/JSON contracts.
The app shows masked status from:

```text
Database Finanziario/catalog/credential_status_masked.csv
Database Finanziario/API/
```

Use runtime environment variables for actual API calls:

```bash
export FRED_API_KEY=...
export ALPHA_VANTAGE_API_KEY=...
```

## Recommended Commands

```bash
python scripts/sync_prices.py --execute --universes sp500,ftsemib --max-symbols 25
python scripts/sync_fundamentals.py --execute --universes sp500,ftsemib --max-symbols 10
python scripts/initial_setup.py --factors ff --execute --max-items 4
```
