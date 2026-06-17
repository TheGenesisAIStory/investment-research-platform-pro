# Macro Context Layer

This layer connects the 95-instrument Macro DB to the equity ML workflow without
turning macro proxies into direct trade decisions.

## Scope

Inputs come from `Database Finanziario/MarketData/Macro` and the lightweight
tables in `output/macro_market/tables`:

- equity risk: SPY
- dollar: DXY
- energy: Brent / WTI
- duration and credit: TLT, SHY, HYG, LQD
- volatility: VIX
- rates/curve proxy: TNX minus IRX
- safe haven / crypto context: GLD, BTC

## Feature Construction

`research_platform_core.macro_context` builds `MacroContextPanel.csv`.
Features are lagged by one observation before being merged into the equity
factor panel. This keeps the block point-in-time conservative.

Core columns:

- `macro_spy_ret21d`, `macro_spy_ret63d`
- `macro_dxy_ret21d`, `macro_dxy_ret63d`
- `macro_brent_ret21d`, `macro_brent_ret63d`
- `macro_wti_ret21d`, `macro_wti_ret63d`
- `macro_tlt_ret21d`, `macro_tlt_ret63d`
- `macro_credit_hyg_tlt_ret63d`
- `macro_credit_lqd_tlt_ret63d`
- `macro_curve_tnx_irx`
- `macro_vix_level`, `macro_vix_change21d`
- `macro_risk_on_score`, `macro_context_score`

## ML Usage

`ml_stock_lab.factor_registry` exposes an experimental block:

```text
macro_context
```

It is optional. The default ML configuration still uses the equity factor
vocabulary; the researcher must explicitly select `macro_context` in ML Stock
Lab to test whether cross-asset context improves out-of-sample IC/RankIC,
Sharpe or hit ratio.

## Regime Detection

`research_platform_core.regime_detection` writes:

- `output/macro_market/tables/MarketRegimeHistory.csv`
- `output/macro_market/tables/MarketRegimeLatest.csv`

Regime labels are intentionally simple and explainable:

- `risk_on`
- `risk_off`
- `crisis`
- `recovery`
- `neutral`

The current regime is shown in Home, Macro View and Portfolio as context only.
It does not modify rankings, weights or model outputs.

## Smart Money Link

CFTC COT positioning is managed separately in `research_platform_core.smart_money`.
The COT snapshot can be used later as a distinct `smart_money` feature block,
but v1 keeps it as a visible context layer until source coverage is stable.

