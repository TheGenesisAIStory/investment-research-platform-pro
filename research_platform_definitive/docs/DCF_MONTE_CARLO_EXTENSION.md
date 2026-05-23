# Model-Based Valuation Uncertainty Extension

This extension imports useful workflow ideas from lightweight DCF tools without creating a separate project. DCF is the pilot model, with Residual Income Monte Carlo and EVA scenario bands now using the same artifact-first pattern.

## Ideas adopted

- Clear workflow: company inputs -> assumptions -> simulation -> results.
- Region-based market parameters for WACC assumptions.
- Monte Carlo on growth, WACC, cost of equity, ROE and terminal growth where model inputs support it.
- Results shown as summary, scenario table/bands, distribution and model-based factors.

## What was adapted

- Data source remains the Research Platform: valuation artifacts, Database Finanziario and notebook memory frames.
- The DCF simulation is a reusable uncertainty layer, not a replacement for the valuation engine.
- Outputs are written as standard artifacts in `company_valuation/output/tables` or the configured `TABLESDIR`.

## Artifact contracts

- `DCFMonteCarloSummary.csv`
- `DCFMonteCarloSimulations.csv`
- `DCFMonteCarloScenarios.csv`
- `DCFMonteCarloAssumptions.csv`
- `DCFMarketParameters.csv`
- `DCFModelBasedFactors.csv`
- `ResidualIncomeMonteCarloSummary.csv`
- `ResidualIncomeMonteCarloSimulations.csv`
- `ResidualIncomeMonteCarloScenarios.csv`
- `ResidualIncomeMonteCarloAssumptions.csv`
- `ResidualIncomeModelBasedFactors.csv`
- `EVAScenarioBands.csv`
- `EVAScenarioSummary.csv`
- `EVAScenarioAssumptions.csv`
- `EVAModelBasedFactors.csv`
- `ModelBasedFactors.csv`

## Model-based factor pattern

`DCFModelBasedFactors.csv`, `ResidualIncomeModelBasedFactors.csv` and `EVAModelBasedFactors.csv` implement the generic model-based factor pattern:

- `dcf_mispricing`
- `dcf_scenario_spread`
- `dcf_probability_undervalued`
- `residual_income_mispricing`
- `residual_income_scenario_spread`
- `residual_income_probability_undervalued`
- `eva_value_gap`
- `eva_scenario_spread`
- `model_factor_family = model_based_mispricing`

ML Stock Lab now searches for these factor artifacts and merges them into the panel/signals by ticker when available. The company screener also exposes a DCF mispricing preset.

## What was not implemented

- No new ingestion layer.
- No vendor-specific beta fetcher.
- No standalone DCF Streamlit project.
- No overwrite of existing valuation models.

## Extension path

1. Promote `ModelBasedFactors.csv` to a documented API-ready contract.
2. Add residual income/EVA factors to the app-level Artifacts registry.
3. Add model-family selection to ML Stock Lab experiments.
4. Add portfolio optimizer penalties for excessive scenario spread.
