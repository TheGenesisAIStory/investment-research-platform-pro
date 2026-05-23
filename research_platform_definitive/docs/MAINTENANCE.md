# Maintenance

## Daily

```bash
python scripts/sync_prices.py --mode incremental --execute
python scripts/validate_data.py
```

## Weekly

```bash
python scripts/sync_fundamentals.py --execute
python research_platform_app/scheduler.py --once --jobs data_platform_status_refresh data_center_validation
```

## Monthly

```bash
python scripts/initial_setup.py --factors ff,aqr,risk --max-items 0 --execute
python scripts/validate_data.py
```

## Scheduler

Template:

```text
config/scheduler_config.yaml
```

## Data Quality Checks

Review:

```text
output/data_quality/DataCenter_target_summary.csv
output/data_quality/DataCenter_stale_inventory.csv
```

Critical watches:

- target coverage below expectations;
- stale prices older than 7 calendar days;
- stale factors older than 31 days;
- provider health checks failing;
- Drive storage pressure.

## Recovery

Regenerate contracts:

```bash
python scripts/migrate_data_api_control.py
python scripts/validate_data.py
```

The large data folders live outside git in `Database Finanziario`; only contracts, manifests and lightweight reports belong in the repo.
