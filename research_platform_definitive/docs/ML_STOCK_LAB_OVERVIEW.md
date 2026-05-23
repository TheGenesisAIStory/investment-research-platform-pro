# ML Stock Lab

`ml_stock_lab` is the repo-native ML equity research library for fair-value estimation, mispricing signals, screening, prediction and quintile portfolio diagnostics.

## Modules

- `datasets.py`: Drive-first panel loading from Database Finanziario, existing valuation/portfolio artifacts and optional AQR factor panels.
- `features.py`: fundamental and technical feature normalization.
- `valuation.py`: sklearn-like fair-value estimators: OLS, LASSO, RF, GBRT, ensemble.
- `signals.py`: absolute/relative mispricing and cross-sectional z-scores.
- `screening.py`: ranking, quantiles, top/bottom stock screening.
- `prediction.py`: expected-return models and temporal train/test split.
- `portfolio.py`: quantile portfolios, long-short weights, forecast-based weights.
- `evaluation.py`: OOS R2, Sharpe, turnover, transaction-cost-adjusted returns and factor alpha.
- `lab.py`: one-shot artifact runner for notebooks, Streamlit and orchestration.

## Core formulas

- Fair value: `V_hat = f_t(X)`
- Relative mispricing: `(V_hat - V_mkt) / V_mkt`
- Cross-sectional z-score: `(MP - mean_t(MP)) / std_t(MP)`
- Long-short quintile: `R_Q5 - R_Q1`
- OOS R2: `1 - SSE_model / SSE_benchmark`

## Data

The library uses the existing platform order:

1. Database Finanziario / Google Drive.
2. Existing notebook artifacts from company valuation, portfolio and screeners.
3. Local cache/output artifacts.
4. API enrichment only when explicitly added later.

## Notebook integration

Main notebooks include a bridge section that calls:

```python
from src.ml_stock_lab import run_ml_stock_lab_experiment
run_ml_stock_lab_experiment(PROJECT_ROOT / "output" / "ml_stock_lab")
```

The dedicated Colab-first notebook is:

`machine_learning_lab/notebooks/ML_Stock_Lab_Experiments.ipynb`

It includes:

- mandatory Colab/local bootstrap as the first code cell,
- a fintech-style widget control center for universe, tickers, model, target and feature blocks,
- paste-friendly API key entry,
- optional AQR factor overlays through `load_aqr_factor_panel`,
- one-click execution via `run_ml_stock_lab_experiment`,
- Plotly visual diagnostics and export browsing.

The human-friendly notebook index is:

`notebooks/README.md`

To avoid duplicate source notebooks, the ML Stock Lab notebook is kept only in its canonical execution path.

## Streamlit integration

The app page is:

`research_platform_app/pages/9_ML_Stock_Lab.py`

It reuses `output/ml_stock_lab/tables/MLStockLab_*` artifacts when available and recalculates only on demand.

## Orchestration

Job id:

`ml_stock_lab_experiments_refresh`

Expected artifacts:

- `MLStockLab_panel.csv`
- `MLStockLab_signals.csv`
- `MLStockLab_metrics.csv`
- `MLStockLab_quintile_returns.csv`
- `MLStockLab_quintile_metrics.csv`
- `MLStockLab_prediction_metrics.csv`
- `MLStockLab_long_short_diagnostics.csv`
- `MLStockLab_status.csv`
- `MLStockLab_experiment_card.csv`

## Caveats

- Fair-value ML is a relative valuation research layer, not a DCF replacement.
- Predictive claims require time-aware splits and out-of-sample evaluation.
- Quintile backtests need liquidity, turnover, factor and transaction-cost checks before production use.
- Missing panel data are surfaced through empty artifacts and metrics, not fabricated.
