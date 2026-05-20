"""Small shared core for notebook-first research platform modules."""

from .data_utils import as_df, first_available, normalize_name, normalize_ticker, resolve_alias
from .data_platform import (
    build_dataset_inventory,
    dataset_status,
    discover_financial_database_root,
    incremental_merge,
    load_catalog_tables,
    provider_fallback_plan,
    publish_artifacts_to_financial_db,
    read_dataset,
    read_dataset_drive_first,
    refresh_europe_stoxx_prices_incremental,
    resolve_data_platform_roots,
    resolve_dataset_path,
    should_refresh,
    summarize_inventory,
    write_data_platform_status,
    write_dataset_incremental,
)
from .export_utils import output_root_from_namespace, safe_write_csv, safe_write_json
from .html_utils import table_html

__all__ = [
    "as_df",
    "first_available",
    "normalize_name",
    "normalize_ticker",
    "resolve_alias",
    "build_dataset_inventory",
    "dataset_status",
    "discover_financial_database_root",
    "incremental_merge",
    "load_catalog_tables",
    "provider_fallback_plan",
    "publish_artifacts_to_financial_db",
    "read_dataset",
    "read_dataset_drive_first",
    "refresh_europe_stoxx_prices_incremental",
    "resolve_data_platform_roots",
    "resolve_dataset_path",
    "should_refresh",
    "summarize_inventory",
    "write_data_platform_status",
    "write_dataset_incremental",
    "output_root_from_namespace",
    "safe_write_csv",
    "safe_write_json",
    "table_html",
]
