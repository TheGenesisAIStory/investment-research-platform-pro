# Gen.is.IA — Academic Feature Reference

## 1. Introduzione metodologica

Gen.is.IA separa tre famiglie di segnali:

- **Cross-sectional equity factors**: value, momentum, quality, size, low risk, liquidity, growth and model-based signals used by Screener and ML Stock Lab.
- **Multi-asset context**: FX, commodities, ETF, fixed-income, crypto and regime features used as optional context, not as hidden trading rules.
- **Portfolio and valuation diagnostics**: risk-adjusted metrics, drawdowns, DCF/WACC, DDM, EVA and comparable-company multiples.

### 1.1 Factor Zoo

The platform uses a conservative subset of the equity factor zoo: signals with long academic history, interpretable formulas and clear point-in-time requirements. New signals are marked experimental until they have stable coverage and out-of-sample monitoring.

### 1.2 Cross-Sectional Predictability

The main ML target family is `forward_return_21d`, `forward_return_63d` and `forward_return_252d`. Model quality is evaluated with IC, RankIC, hit ratio, long-short spread and portfolio Sharpe. These metrics answer whether the model ranks securities well at each date.

### 1.3 Point-In-Time Safety

Price-only features can use values known at date `t`. Fundamental features require reporting lags, typically one quarter or six months depending on the statement item. Target columns are never valid model inputs.

### 1.4 Winsorization and Normalization

The recommended default is cross-sectional winsorization at 1%/99%, then rank or z-score normalization by date and universe. This avoids one extreme outlier dominating a factor composite.

### 1.5 Regime Dependency

Macro and time-series signals are used as context for interpretation and monitoring. They may be promoted to ML feature blocks only when tests show stable incremental IC and no leakage.

## 2. Fattori accademici fondamentali

| Family | Factors | Source | Platform status |
| --- | --- | --- | --- |
| Fama-French 3-Factor | MKT-RF, SMB, HML, RF | Ken French Data Library | US, EU, JP, APAC, EM, World download helpers |
| Fama-French 5-Factor | MKT-RF, SMB, HML, RMW, CMA, RF | Ken French Data Library | US, EU, JP, APAC download helpers |
| Carhart Momentum | MOM/UMD | Ken French Data Library | US, EU, JP, APAC download helpers |
| Italy local factors | MKT_RF, SMB, HML, MOM, RF | Local factor panel construction | Proxy from Italian tickers where available |
| AQR factor library | QMJ, BAB, HML Devil and related datasets | AQR Data Library | Discovery/cache provider |

## 3. Feature equity implementate

| Feature | Formula | Paper / rationale | PIT status |
| --- | --- | --- | --- |
| `momentum_12m_1m` | `$MOM_t = \\frac{P_{t-21}-P_{t-252}}{P_{t-252}}$` | Jegadeesh & Titman (1993); skip last month to reduce short-term reversal noise | price-only, safe at `t` |
| `momentum_reversal_1m` | `$REV_t = -\\frac{P_t-P_{t-21}}{P_{t-21}}$` | Jegadeesh (1990), De Bondt & Thaler (1985) | price-only, safe at `t` |
| `book_to_market` | `$B/M_t = \\frac{BookEquity_{t-lag}}{MarketCap_t}$` | Fama & French (1992, 1993) | requires fundamental lag |
| `earnings_yield` | `$E/P_t = \\frac{EPS_{TTM,t-lag}}{P_t}$` | Basu (1977) | requires earnings lag |
| `fcf_yield` | `$FCF/MC_t = \\frac{FCF_{TTM,t-lag}}{MarketCap_t}$` | Lakonishok, Shleifer & Vishny (1994) | requires cash-flow lag |
| `gross_profitability` | `$GPA_t = \\frac{Revenue_t-COGS_t}{TotalAssets_t}$` | Novy-Marx (2013) | requires statement lag |
| `roe` | `$ROE_t = \\frac{NetIncome_{TTM}}{AvgBookEquity}$` | Piotroski (2000), quality/profitability literature | requires statement lag |
| `roic` | `$ROIC_t = \\frac{EBIT_t(1-tax)}{Equity+Debt-Cash}$` | Penman (2001), Koller et al. | requires statement lag |
| `accruals_ratio` | `$ACC_t = \\frac{NI_t-CFO_t}{AvgTotalAssets_t}$` | Sloan (1996) | requires statement lag |
| `piotroski_f_score` | `$F=\\sum_{i=1}^{9} F_i$` | Piotroski (2000) | requires comparable prior statements |
| `altman_z_score` | `$Z'=6.56X_1+3.26X_2+6.72X_3+1.05X_4$` | Altman (1968, 1995 revised) | requires statement lag |
| `log_market_cap` | `$SIZE_t=\\ln(P_t \\times Shares_t)$` | Banz (1981), Fama & French (1992) | safe when shares are lagged |
| `beta_market` | `$\\beta_i=\\frac{Cov(r_i,r_m)}{Var(r_m)}$` | Sharpe (1964) | rolling returns, safe at `t` |
| `idiosyncratic_vol` | `$IVOL_i=std(\\epsilon_i)$` | Ang et al. (2006) | rolling residuals only |
| `amihud_illiquidity` | `$ILLIQ=\\frac{1}{T}\\sum \\frac{|r_d|}{DollarVolume_d}\\times10^6$` | Amihud (2002) | rolling price/volume |
| `asset_growth` | `$AG_t=\\frac{TA_t-TA_{t-1}}{TA_{t-1}}$` | Cooper, Gulen & Schill (2008) | requires statement lag |

## 4. Portfolio Analytics

| Metric | Formula | Interpretation |
| --- | --- | --- |
| Sharpe | `$\\frac{E[R_p-R_f]}{\\sigma_p}$` | excess return per unit volatility |
| Sortino | `$\\frac{E[R_p-R_f]}{\\sigma_{down}}$` | downside-risk-adjusted return |
| Information Ratio | `$\\frac{R_p-R_b}{\\sigma(R_p-R_b)}$` | active return per unit tracking error |
| Calmar | `$CAGR / |MaxDrawdown|$` | return per unit worst path loss |
| Omega | `$\\frac{\\int_T^\\infty (1-F(r))dr}{\\int_{-\\infty}^T F(r)dr}$` | gain/loss balance around a threshold |
| Ulcer Index | `$\\sqrt{mean(drawdown_t^2)}$` | depth and persistence of underwater periods |
| CVaR | `$E[R \\mid R \\le VaR_\\alpha]$` | expected shortfall in the tail |
| Brinson allocation | `$(w_p-w_b)(r_{b,s}-r_b)$` | sector allocation contribution |
| Brinson selection | `$w_b(r_{p,s}-r_{b,s})$` | stock selection contribution |

## 5. Valuation Models

| Model | Formula | Notes |
| --- | --- | --- |
| DCF | `$EV=\\sum_{t=1}^{N}\\frac{FCF_t}{(1+WACC)^t}+\\frac{TV}{(1+WACC)^N}$` | terminal value uses Gordon growth |
| WACC | `$(E/V)R_e+(D/V)R_d(1-tax)$` | CAPM equity cost and after-tax debt cost |
| DDM | `$P_0=\\frac{D_1}{R_e-g}$` | only meaningful for dividend payers |
| EVA | `$NOPAT-WACC\\times InvestedCapital$` | value creation after capital charge |
| Comps | `implied price from sector median multiples` | compare PE, EV/EBITDA, EV/Sales and PB |

## 6. Smart Money Indicators

Smart Money is a context layer. COT positioning, ETF flow proxies and optional put/call ratios are informational diagnostics, not direct model targets.

## 7. Macro Context Features

| Feature | Formula | Use |
| --- | --- | --- |
| `macro_spy_ret21d` | `SPY_t / SPY_t-21 - 1`, lagged one observation | short-horizon global equity risk appetite |
| `macro_spy_ret63d` | `SPY_t / SPY_t-63 - 1`, lagged one observation | medium-horizon equity backdrop |
| `macro_dxy_ret21d` | `DXY_t / DXY_t-21 - 1`, lagged one observation | USD tightening/liquidity pressure |
| `macro_brent_ret21d` | `Brent_t / Brent_t-21 - 1`, lagged one observation | commodity/reflation impulse |
| `macro_tlt_ret21d` | `TLT_t / TLT_t-21 - 1`, lagged one observation | duration and flight-to-quality context |
| `macro_credit_spread` | `return_21d(HYG) - return_21d(TLT)`, lagged one observation | credit risk appetite versus duration |
| `macro_yield_slope` | `US 10Y proxy - US 2Y/front-end proxy`, lagged one observation | curve and rate-cycle context |
| `macro_vix_level` | `rank_pct(VIX over 252d)`, lagged one observation in ML feature construction | volatility stress percentile |
| `macro_regime_encoded` | `risk_on=1, recovery=0.5, risk_off=-0.5, crisis=-1`, lagged one observation | compact regime state for experimental models |

Source data: internal Macro DB, 140 multi-asset proxies, 2000-2026 where available. The ML join uses a minimum one-observation delay and an as-of merge, so an equity row at date `t` cannot see macro observations after `t-1`. These variables are experimental and should be evaluated by incremental IC/RankIC/Sharpe versus the canonical equity factor block.

Related literature: factor timing and conditional expected returns in Fama and French (1989, 1993), credit-cycle information in Adrian and Shin (2010), and volatility/regime conditioning in Ang and Bekaert (2002).

## 8. Market Regime Classification

The regime detector writes `output/macro_market/tables/MarketRegimeHistory.csv` with four labels:

| Regime | Definition | Core rule |
| --- | --- | --- |
| `risk_on` | constructive equity/credit backdrop | positive 21d equity momentum, low volatility, no credit stress |
| `risk_off` | defensive or deteriorating backdrop | VIX warning, negative equity momentum, credit stress or inverted curve |
| `crisis` | acute stress | VIX stress, equity momentum below -5% or 63d below -10%, and credit stress |
| `recovery` | rebound after stress | positive equity momentum after `risk_off` or `crisis`, without crisis-level volatility |

Signals stored with each row: `equity_momentum_21d`, `credit_spread`, `vix_proxy`, `yield_slope`, `commodity_momentum`, plus `confidence` and human-readable `drivers`. If one proxy is missing, the classifier degrades gracefully and scales confidence by signal availability.

Historical distribution is generated from the local Macro DB when `build_regime_history()` or `build_market_regime_history()` runs. The distribution is intentionally not hard-coded in this document because it depends on the data snapshot.

## 9. WorldQuant 101 Formulaic Alphas

The platform implements the 101 formulaic alphas from Kakushadze (2016), arXiv:1601.00991, in `research_platform_core.alpha101`. Each alpha is exposed through `Alpha101Suite`, registered in `feature_metadata.py` as category `alpha101`, and selectable in ML Stock Lab as an experimental feature block.

These alphas are price/volume formulas, cross-sectionally rank-normalized, and computed only when the panel has OHLCV fields. They are not mixed into the canonical academic factor layer by default; the researcher must explicitly select `alpha101` and compare the resulting IC/RankIC/Sharpe against value, quality, momentum, risk, size and growth baselines.

## 10. Legacy Macro Regime Features

| Feature | Formula | Use |
| --- | --- | --- |
| `regime_spy_trend_63d` | `sign(SPY 63d return)` | equity risk appetite |
| `regime_vix_level` | VIX low/mid/high bucket | volatility stress |
| `regime_yield_slope` | `US 10Y proxy - US 2Y/front-end proxy` | curve/rate regime |
| `regime_dxy_trend_21d` | `sign(DXY 21d return)` | USD pressure |
| `regime_gold_trend_21d` | `sign(Gold 21d return)` | defensive/inflation proxy |

## 11. Riferimenti bibliografici

- Adrian, T., Shin, H. S. (2010). Liquidity and leverage.
- Amihud, Y. (2002). Illiquidity and stock returns.
- Ang, A., Bekaert, G. (2002). International asset allocation with regime shifts.
- Ang, A., Hodrick, R., Xing, Y., Zhang, X. (2006). The cross-section of volatility and expected returns.
- Asness, C., Frazzini, A., Pedersen, L. (2019). Quality Minus Junk.
- Banz, R. (1981). The relationship between return and market value of common stocks.
- Basu, S. (1977). Investment performance of common stocks in relation to their price-earnings ratios.
- Carhart, M. (1997). On persistence in mutual fund performance.
- Cooper, M., Gulen, H., Schill, M. (2008). Asset growth and the cross-section of stock returns.
- De Bondt, W., Thaler, R. (1985). Does the stock market overreact?
- Fama, E., French, K. (1992, 1993, 2015). Cross-sectional returns and factor models.
- Frazzini, A., Pedersen, L. (2014). Betting Against Beta.
- Jegadeesh, N., Titman, S. (1993). Returns to buying winners and selling losers.
- Kakushadze, Z. (2016). 101 Formulaic Alphas. arXiv:1601.00991.
- Novy-Marx, R. (2013). The other side of value.
- Piotroski, J. (2000). Value investing and historical financial statement information.
- Sloan, R. (1996). Do stock prices fully reflect information in accruals and cash flows?
