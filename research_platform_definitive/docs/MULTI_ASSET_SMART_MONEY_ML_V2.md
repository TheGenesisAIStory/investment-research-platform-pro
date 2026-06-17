# Gen.is.IA Multi-Asset, Smart Money and ML Governance v2

This note documents the v2 content layer added on top of the v1-ready equity
factor/ML platform.

## Multi-Asset Universe

Backend module: `src/research_platform_core/multi_asset_universe.py`

Artifacts:

- `output/multi_asset_universe/ohlcv/{SYMBOL}.parquet`
- `output/multi_asset_universe/MultiAssetUniverseManifest.csv`
- `output/multi_asset_universe/MultiAssetLatestSnapshot.csv`
- `output/macro_market/tables/MultiAssetUniverseManifest.csv`

Coverage domains:

- FX
- Commodities
- ETF Equity
- ETF Fixed Income
- ETF Commodity
- Crypto
- Fixed Income / yield proxies

The ingestion uses yfinance daily history with `auto_adjust=True`. Each row is
classified as `OK` or `PARTIAL` and includes first/last date, row count, target
path and source metadata.

## Smart Money v1

Backend module: `src/research_platform_core/smart_money.py`

Official and proxy sources:

- CFTC COT positioning for core equity/rates/FX/commodity futures.
- ETF public flow proxy for SPY, QQQ, IWM, GLD, TLT, HYG, EEM, IEUR.
- Optional options put/call ratio proxy for SPY, QQQ and GLD.

Artifacts:

- `output/smart_money/tables/SmartMoney_COT_history.csv`
- `output/smart_money/tables/SmartMoney_COT_snapshot.csv`
- `output/smart_money/cot/{CONTRACT}_cot_history.parquet`
- `output/smart_money/etf_flows/etf_flows_history.parquet`
- `output/smart_money/etf_flows/etf_flows_snapshot.csv`
- `output/smart_money/options/pcr_snapshot.csv`
- `output/smart_money/tables/SmartMoneyAssetCatalog.csv`
- `output/smart_money/tables/SmartMoneySourceManifest.csv`

ETF flows are explicitly labelled as public-data proxies. They should be
replaced by licensed creations/redemptions or provider flow files when available.

## Macro Regime Features

Backend modules:

- `src/research_platform_core/macro_context.py`
- `src/research_platform_core/regime_detection.py`
- `src/ml_stock_lab/factor_registry.py`

Feature block:

- `macro_context`: broad lagged cross-asset returns/spreads.
- `macro_regime`: compact lagged regime flags:
  - `regime_spy_trend_sign`
  - `regime_vix_regime`
  - `regime_yield_curve_slope`
  - `regime_dxy_trend`
  - `regime_gold_trend`

All regime features are lagged one observation before joining into the equity
factor panel.

## Factor Portfolio Baselines

Backend module: `src/research_platform_core/factor_portfolio_baselines.py`

Artifacts:

- `output/factor_baselines/baseline_portfolio_metrics.csv`
- `output/factor_baselines/baseline_portfolio_returns.csv`

Strategies:

- Long-only top decile, equal-weighted, monthly rebalance.
- Long-short top 20% minus bottom 20%, dollar-neutral proxy, monthly rebalance.

Metrics:

- annualized return
- Sharpe
- Sortino
- max drawdown
- mean IC
- turnover

These benchmarks are the control group for OLS/RF/GBRT/ensemble models in ML
Stock Lab.

## Model Monitoring

Backend module: `src/research_platform_core/model_monitoring.py`

Artifacts:

- `output/ml_lab/model_monitoring/model_monitoring_summary.csv`
- `output/ml_lab/model_monitoring/{MODEL}_rolling_ic.parquet`

The monitor computes daily IC/RankIC by model and rolling 12M IC/RankIC. It is
shown in ML Stock Lab under “Model Performance Over Time”.

