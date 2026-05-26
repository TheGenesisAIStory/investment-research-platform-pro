# Changelog

## v2.x - Factor Zoo, EPS, Simulation and Cloud Handoff

- Added institutional/cross-asset factor zoo implementation (`987779a`) with FX, fixed-income, commodity, smart-money and cross-asset momentum blocks.
- Resolved `ML_CONTENT_LAYER_V2.md` cross-asset documentation overlap (`1c849c7`) by keeping the detailed academic treatment in `FEATURE_ACADEMIC_REFERENCE.md`.
- Added experimental EPS factors and Monte Carlo simulation helpers (`223903f`) with point-in-time lag rules and reproducible `seed=42` defaults.
- Added feature-gap audit and Colab ML handoff notebook (`901e7e4`) for cloud-oriented training runs instead of forcing heavy local retraining.
- Implemented experimental `cross_asset_value` and rolling PCA `global_risk_factor` in the existing cross-asset module; `liquidity_factor` remains a documented STUB pending a standardized cross-asset liquidity panel.
- Documented local-only data completion manifest gaps as artifact availability issues, not CI-blocking code regressions.
