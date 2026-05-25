# Gen.is.IA Feature Completeness V3

This note documents the v3 content layer added on top of the v1-ready research
workstation. It is intentionally additive: existing factor panels and artifacts
remain valid, while new metrics/features are optional and surfaced through
metadata, UI controls and tests.

## Portfolio Analytics

`research_platform_core.portfolio_analytics` adds a broader portfolio analytics
toolkit:

- Risk-adjusted performance: Sharpe, Sortino, Calmar, Omega, Kappa,
  Information Ratio, Treynor, Jensen alpha and M-squared.
- Drawdown depth and duration: maximum drawdown, average drawdown, average and
  maximum drawdown duration, Ulcer Index, Pain Index and Recovery Factor.
- Distribution and tail metrics: skewness, kurtosis, parametric VaR 95/99,
  CVaR 95/99, Tail Ratio, Hit Rate and average up/down period returns.
- Implementation metrics: annual turnover, implied holding period, estimated
  cost drag and net Sharpe.
- Construction: equal weight, minimum variance, maximum Sharpe, inverse-vol risk
  parity and efficient frontier. Optimizers use `scipy.optimize` with
  long-only constraints and ridge-regularized covariance.
- Attribution: a compact Brinson-style allocation / selection / interaction
  decomposition when sector, return and benchmark weights are available.

The Portfolio page now exposes these in `Risk Analytics`, `Portfolio
Construction` and `Attribution` tabs. They are diagnostics only; optimizer
weights do not replace the portfolio artifact.

## Valuation Analytics

`research_platform_core.valuation_analytics` provides reusable valuation helpers:

- WACC via CAPM cost of equity and after-tax cost of debt.
- DCF with explicit stage-1 growth, terminal growth, projected FCF, terminal
  value, enterprise value, equity value, intrinsic price and upside.
- Market multiples: P/E, P/B, P/S, P/CF, EV/EBITDA, EV/EBIT, EV/Sales, EV/FCF,
  dividend yield, payout ratio and PEG.
- EVA / residual-income diagnostics: NOPAT, invested capital, ROIC, WACC spread,
  EVA and intrinsic P/B.
- Dividend Discount Model: Gordon and two-stage DDM.
- Asset-based valuation: book value per share, tangible book, NAV and a
  conservative liquidation proxy.
- Comparable company analysis: sector median P/E and EV/EBITDA with implied
  prices and premium/discount.

The Valuation page now has explicit tabs for `DCF & Intrinsic Value`, `Multipli
di Mercato`, `Residual Income / EVA` and `DDM & Fair Value Summary`, in addition
to the existing Monte Carlo/scenario workflow.

## Equity Feature Engineering

`research_platform_core.equity_feature_engineering` adds optional course-style
features:

- Technical advanced: 52-week high/low proximity, 12-1M momentum,
  short-term reversal, volatility term structure, downside volatility,
  Amihud illiquidity, volume ratio, RSI, MACD, Bollinger band position,
  SMA50/SMA200 ratios and golden cross.
- Quality advanced: ROA, ROIC, margins, liquidity ratios, leverage ratios,
  interest coverage, Altman Z-score, accruals ratio and Piotroski F-Score.
- Value advanced: book-to-market, earnings yield, FCF yield, EBITDA yield,
  sales-to-price, EV multiples and PEG.
- Growth advanced: 1Y/3Y revenue and earnings growth, FCF growth, capex
  intensity, R&D intensity and asset growth.

The factor registry now includes four optional experimental blocks:

- `technical_advanced`
- `quality_advanced`
- `valuation_advanced`
- `growth_advanced`

These blocks are selectable in ML Stock Lab and filterable in Screener when the
columns exist in the current artifacts. The feature metadata registry documents
formula, interpretation, sub-category, data requirements and source paper where
applicable, including Fama-French, Jegadeesh-Titman, Amihud, Piotroski, Sloan
and Altman references.

## Anti-Leakage Notes

- Price features are computed from current or trailing observations only.
- Forward-return targets remain excluded by the leakage denylist in
  `ml_stock_lab.factor_registry`.
- Piotroski uses current and prior ticker observations. Production refreshes
  should continue to use point-in-time fundamentals with the existing lag policy.
- Time-series and macro-context features are optional and labelled experimental.

## Tests

`tests/test_feature_portfolio_valuation_v3.py` covers:

- Piotroski F-Score criteria.
- Advanced technical feature generation.
- Advanced fundamental feature generation.
- WACC and DCF.
- Multiples, EVA, DDM and asset-based valuation.
- Sector comps.
- Advanced portfolio metrics.
- Portfolio optimizers and efficient frontier.
