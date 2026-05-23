# Laboratorio Colab Review Checklist

Use this checklist before promoting any reference notebook from `laboratorio/` into the canonical platform.

## Mandatory checks

- Open the notebook from `research_platform_definitive/laboratorio/notebooks_ml4t_reference/`.
- Run the canonical Colab setup cell from the platform notebooks before importing local modules.
- Confirm it does not write to `/content/...` when running locally outside Colab.
- Confirm all writes go to a runtime cache, `output/`, or Database Finanziario when explicitly enabled.
- Confirm optional dependencies are guarded with clear install notes.
- Confirm no paid API/provider is called unless the user opted in and configured a key.
- Confirm the notebook produces an artifact that can feed one platform target:
  - `ml_stock_lab`
  - `portfolio_research`
  - `valuation_research`
  - `smart_money_engine`
  - `data_platform`
  - `future_microstructure_lab`

## Promotion rule

Do not expose a reference notebook in the app as a runnable workflow until the useful logic has been extracted into `src/` or wrapped behind an artifact contract.

## Current status

The curated notebooks are JSON-valid and inventoried. They have not all been executed end-to-end in Colab because many are optional references with external dependencies or large data assumptions.
