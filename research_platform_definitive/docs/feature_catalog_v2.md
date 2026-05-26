# Gen.is.IA Feature Catalog v2

This release harmonizes the research metadata layer around:

- 188+ registered features with formula, LaTeX formula, source, DOI/URL, economic rationale, factor-zoo category and point-in-time flags.
- 79+ registered metrics with formula, LaTeX formula, source, interpretation range, higher-is-better flag and metric family.
- New factor families: `factor_alpha`, `factor_investment`, extended profitability, liquidity, macro regime and smart-money/sentiment signals.
- Multi-provider fundamentals waterfall: FMP, Polygon, EODHD, Intrinio and yfinance fallback.
- Robust options PCR context through CBOE broad put/call history plus Polygon/FMP extension hooks.
- Italy local factor scaffolding for FTSE MIB names, BTP-Bund spread and Eurostat/ECB macro context.

## Point-In-Time Safety

All new investment and accounting-derived metadata entries declare a reporting lag through `lag_required`.
The implemented `compute_investment_factors()` shifts computed factors by one reporting observation per ticker and annotates `pit_lag_days = 45`.
Targets such as forward returns are marked point-in-time safe as labels, not live predictors.

## UI

The new Streamlit page `11_📖_Glossary.py` exposes feature and metric registries with filters for category, factor-zoo category, source paper and asset class.
`15_📚_Academic_Reference.py` remains the deeper academic factor browser.

## Provider Notes

API-key-backed providers are optional. Missing keys return `None`, empty frames or `PARTIAL` status envelopes instead of crashing app startup or tests.
Configured keys are read from environment variables or the orchestrator `api_keys` argument.
