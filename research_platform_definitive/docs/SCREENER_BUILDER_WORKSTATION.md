# Screener Builder Workstation

## UI Design

`Screener_Builder` is the buy-side idea generation cockpit. The page is split into two panels:

- Left panel: saved screeners, templates, universe filters, fundamental filters, ML/Smart Money filters, sorting and save controls.
- Right panel: results table, cross-page actions, diagnostics and explanation tabs.

The top context bar shows shared state:

- reporting currency,
- benchmark,
- selected ticker,
- active screener.

Shared state is stored in `st.session_state`:

- `selected_ticker`,
- `active_screener_config`,
- `active_screener_name`,
- `preferred_screening_columns`,
- `last_screening_result_count`.

## Unified Data Model

The workbench builds one screening frame from:

- Company Valuation screener and valuation artifacts,
- Portfolio Selection artifacts,
- ML Stock Lab signals,
- Smart Money / Gov Data scores and event feed,
- local equity metadata when available.

The merge key is normalized `ticker`. Missing data is surfaced as empty/null values rather than imputed silently.

Core display fields include:

- issuer identity: ticker, company name, sector, industry, country, universe,
- fundamentals: P/E, EV/EBITDA, P/B, dividend yield, ROE, leverage,
- model fields: ML score, ML quintile, fair-value proxy, mispricing,
- Smart Money fields: composite score, event count, coverage note,
- portfolio fields: selection score, weight,
- freshness fields: fundamentals, ML, Smart Money update markers.

## Presets

The initial templates are:

- Quality at reasonable price,
- Dividend compounders,
- ML high conviction,
- Value + improving momentum,
- Smart Money confirmed.

Templates are intentionally editable. Applying a template seeds the UI, then the analyst can adjust filters and save the final version.

## Saved Screeners

Saved screeners are lightweight JSON artifacts under:

```text
output/screeners/
```

Each file stores:

- name,
- description,
- timestamp,
- result count at save,
- filter/sort configuration.

The UI can list, load and delete saved screeners. Current result count is recomputed when the page loads.

## Reasoning Panels

The explanation tabs distinguish real attribution from proxy attribution:

- ML reasoning uses SHAP-like proxy drivers when local feature attribution tables are missing.
- Smart Money reasoning shows issuer events and coverage notes.
- Valuation summary shows available multiple/fair-value drivers and sends the user to Valuation Research for full DCF assumptions.

Future versions should attach formal feature attribution artifacts from ML Stock Lab and DCF driver contribution tables from Valuation Research.

## Future Hardening

- Add user identity and permissions for personal vs team screeners.
- Add audit logging for save/load/delete and notebook refresh actions.
- Store screener configs in SQLite/Postgres when multi-user collaboration starts.
- Add row-level action menus once Streamlit supports richer dataframe callbacks.
- Add explicit freshness SLA badges for fundamentals, ML and Smart Money artifacts.
