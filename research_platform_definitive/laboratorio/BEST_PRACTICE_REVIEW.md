# Laboratorio Research Library - Best Practice Review

Generated: 2026-05-20 18:56

This folder is no longer a raw dump of old notebooks. It is a curated research library for the definitive Research Platform.

## Selection rule

Kept notebooks must add practical value to at least one canonical platform area:

- Data Platform / provider registry / storage benchmarks
- ML Stock Lab / factor research / walk-forward validation
- Portfolio Research / risk, allocation, backtesting and tear-sheet reporting
- Company Valuation / fundamentals, SEC, text and event intelligence
- Smart Money Government Data Engine / regulatory and alternative data
- Optional future microstructure lab backed by local ITCH data preserved outside Drive

Generic teaching notebooks, image/NLP toy examples, broad deep-learning intros, GAN/DRL demos and raw dataset builders unrelated to the platform were removed from this definitive bundle.

## Result

- Curated notebooks kept: **75**
- Files kept including support files: **129**
- Files removed from definitive laboratorio bundle: **153**

## Best-practice repository mapping

See `BEST_PRACTICE_REPOSITORIES.csv` for the source-to-platform mapping. The key external patterns reviewed were:

- `stefan-jansen/machine-learning-for-trading`: notebook-first quant research patterns
- `cloudQuant/alphalens`: factor diagnostics and quantile evaluation
- `ranaroussi/quantstats`: portfolio analytics and HTML tear sheets
- `dcajasn/Riskfolio-Lib` and `skfolio/skfolio`: risk-aware portfolio optimization APIs
- `microsoft/qlib`: production-style quant research architecture
- `pmorissette/bt`: composable backtesting
- `AI4Finance-Foundation/FinRL`: future optional RL environment reference, not promoted now

## App integration

The Streamlit app exposes this inventory through the `Laboratorio Research Library` page. Notebooks remain references until promoted into `src/`, canonical notebooks, or app pages.
