# Notebook Review: Paper-1 TCEL vs ECL

## Executive summary
The notebook was rebuilt as a reproducible research asset for the paper *TTC Clean Expected Loss (TCEL) as a Management Metric for Pricing, EVA and Economic Capital*. It now supports the paper with a full empirical pipeline: data provenance, TCEL/ECL construction, EVA/RAROC diagnostics, regressions, walk-forward ML, ablation, interpretability, robustness and dashboard reporting.

## Critical issues
- The original notebook skeleton used emoji-prefixed H2 headers; these were normalized to the mandatory 0-14 section titles.
- The notebook now treats `Paper 1 su TTC Clean Expected Loss (TCEL).ipynb` as the primary asset and does not rely on Company Valuation as a target.
- Real portfolio-level TCEL/ECL files were not found, so the notebook logs a warning and creates realistic synthetic data with `_synthetic` source columns.

## Methodological issues
- All predictive features are shifted by one quarter before modelling.
- Walk-forward validation is chronological with an embargo and train-only scaling.
- TCEL, ECL, macro and staging blocks are tested separately through ablation.

## Code quality issues
- Added central `EXPERIMENT`, `PORTFOLIOS`, `COLORS`, output directories and reusable save/register helpers.
- All figures and tables are written to deterministic local paths.
- Logging now uses both `FileHandler` and `StreamHandler`.

## Reproducibility issues
- Synthetic fallback is seeded with `42`.
- Tables include `N`; figures are saved before display at 180 dpi.
- The artifact manifest records all generated deliverables.

## Visualization/reporting issues
- Added the ten required paper figures plus a standalone HTML research dashboard.
- Tables and figures use consistent naming and relative dashboard paths.

## Changes applied
- Completed all 15 notebook sections.
- Generated required Tables 1-19 and Figures 1-10.
- Added final conclusion, table summary, dashboard and artifact manifest.

## Remaining limitations
- Results based on synthetic data are methodological evidence, not confidential bank empirical evidence.
- Production use should replace the synthetic panel with audited portfolio-level TCEL/ECL/EAD/EC data.
- The econometric design is intentionally transparent; future versions can add clustered standard errors and richer dynamic panel specifications.
