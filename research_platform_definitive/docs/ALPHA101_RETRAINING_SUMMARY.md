# Alpha101 Retraining Summary

Date: 2026-05-26

Command:

```bash
python research_platform_definitive/scripts/train_ml_models_2000_2026.py \
  --models ols,rf,gbrt,ensemble \
  --target-horizon-days 21 \
  --feature-blocks value,quality,momentum,risk,size,growth,alpha101 \
  --use-alpha101 True \
  --max-rows 500000
```

Status: `OK`

Artifacts:

- `output/ml_training_lab/MLTraining_manifest_alpha101_summary.json`
- `output/ml_training_lab/tables/MLTraining_metrics_alpha101_summary.csv`
- `output/ml_training_lab/tables/MLTraining_metrics_alpha101.csv`
- `output/ml_training_lab/tables/MLTraining_feature_importance_alpha101.csv`
- `output/ml_training_lab/tables/MLTraining_model_cards_alpha101.csv`
- `output/ml_training_lab/tables/MLTraining_predictions_alpha101.csv`
- `output/ml_training_lab/tables/MLTraining_predictions_wide_alpha101.csv`
- `output/ml_training_lab/tables/MLTraining_panel_alpha101.csv`
- `output/ml_stock_lab/tables/MLStockLab_model_comparison_alpha101.csv`
- `output/ml_stock_lab/tables/MLStockLab_trained_model_signals_alpha101.csv`

The full panel and prediction artifacts are intentionally kept local because
they are large. The committed documentation captures the reproducible run
configuration and the model metrics.

## Run Notes

- Split: train 2000-01-03 to 2018-12-31, test 2019-01-02 to 2026-05-22.
- Target: `forward_return`, 21 trading days.
- Feature blocks: `value,quality,momentum,risk,size,growth,alpha101`.
- Feature count: 121.
- Panel rows after feature/target filtering: 275,000.
- Test predictions: 137,220 per model.
- Tree models use a deterministic 15,000-row fit cap for local Alpha101 runs.
- The factor panel used by this job contains `price` and `market_value`, but not
  full OHLCV. For this bounded retraining the ML pipeline used the Alpha101
  rank-proxy fallback; the complete Kakushadze formulas remain available in
  `research_platform_core.alpha101` for OHLCV panels.

## Metrics

| Model | Status | IC | RankIC | Long-short Sharpe | Net long-short Sharpe | Fit rows | Prediction rows |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OLS | OK | 0.0175 | -0.0451 | 0.6212 | 0.6208 | 137,780 | 137,220 |
| RF | OK | -0.0102 | -0.0085 | 0.6202 | 0.6199 | 15,000 | 137,220 |
| GBRT | OK | -0.0117 | -0.0185 | 0.5969 | 0.5966 | 15,000 | 137,220 |
| Ensemble | OK | -0.0057 | -0.0272 | 0.6214 | 0.6210 | 15,000 | 137,220 |
