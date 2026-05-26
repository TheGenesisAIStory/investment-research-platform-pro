# Gen.is.IA Feature Gap Report

Generated for branch `codex/research-platform-v2-ml-ollama`.

Decision policy applied:

- `ML_CONTENT_LAYER_V2.md`: `docs_only`; removed cross-asset content was already covered in `FEATURE_ACADEMIC_REFERENCE.md`.
- EPS factors: `create_new_module`; implemented as experimental in `factors/eps_factors.py`.
- Monte Carlo simulation: `create_new_module`; implemented as experimental in `simulation/monte_carlo.py`.
- Colab Pro pipeline: `cloud_handoff`; created notebook skeleton in `colab/ml_training_pipeline.ipynb`.
- Cross-asset completeness check: `extend_existing_module`; `cross_asset_value` and `global_risk_factor` are now implemented in the existing module, while `liquidity_factor` remains a documented `STUB`.

## Academic Documentation Audit

| Module | Factor | Reference | Status | Missing Doc? |
| --- | --- | --- | --- | --- |
| `institutional_factors.py` | Short-term reversal | Jegadeesh 1990; Bartram et al. 2021 | implemented | no |
| `institutional_factors.py` | Gross profitability | Novy-Marx 2013, Journal of Financial Economics | implemented | no |
| `institutional_factors.py` | Investment / CMA proxy | Fama-French 2015, Journal of Financial Economics | implemented | no |
| `institutional_factors.py` | Accruals | Sloan 1996, The Accounting Review | implemented | no |
| `institutional_factors.py` | Cash profitability | Ball et al. 2016, Journal of Financial Economics | implemented | no |
| `institutional_factors.py` | Earnings quality | Bartram et al. 2021, Journal of Business Economics | implemented | no |
| `institutional_factors.py` | Net payout yield | Boudoukh et al. 2007, Journal of Finance | implemented | no |
| `institutional_factors.py` | Distress risk | Altman 1968; Bartram et al. 2021 | implemented | no |
| `fx_factors.py` | FX carry proxy | Lustig-Verdelhan 2007, American Economic Review; Bartram et al. 2021 | experimental | no |
| `fx_factors.py` | FX 12-1 momentum | Menkhoff et al. 2012, Journal of Finance | experimental | no |
| `fx_factors.py` | FX trend | Asness-Moskowitz-Pedersen 2013, Journal of Finance | experimental | no |
| `fx_factors.py` | FX volatility | Menkhoff et al. 2012, Journal of Finance | experimental | no |
| `fi_factors.py` | Term carry | Fama-Bliss 1987; Bartram et al. 2021 | experimental | no |
| `fi_factors.py` | Credit carry | Elton et al. 2001, Journal of Finance | experimental | no |
| `fi_factors.py` | FI momentum | Asness-Moskowitz-Pedersen 2013, Journal of Finance | experimental | no |
| `fi_factors.py` | Real-yield proxy | Bartram et al. 2021, Journal of Business Economics | experimental | no |
| `commodity_factors.py` | Commodity momentum | Gorton-Rouwenhorst 2006, Financial Analysts Journal | experimental | no |
| `commodity_factors.py` | Commodity trend | Asness-Moskowitz-Pedersen 2013, Journal of Finance | experimental | no |
| `commodity_factors.py` | Commodity carry proxy | Gorton-Rouwenhorst 2006; Bartram et al. 2021 | experimental | no |
| `commodity_factors.py` | Commodity mean reversion | Asness-Moskowitz-Pedersen 2013, Journal of Finance | experimental | no |
| `cross_asset_factors.py` | cross_asset_momentum | Asness-Moskowitz-Pedersen 2013, Journal of Finance | experimental | no |
| `cross_asset_factors.py` | Cross-asset value | Asness-Moskowitz-Pedersen 2013, Journal of Finance | implemented experimental | no |
| `cross_asset_factors.py` | Global risk factor | Bartram et al. 2021; PCA risk-factor literature | implemented experimental | no |
| `cross_asset_factors.py` | Liquidity factor | Pastor-Stambaugh 2003, Journal of Political Economy; Amihud 2002, Journal of Financial Markets | STUB - owner: Data Platform; timeline: next provider-liquidity cycle | yes - implementation missing |
| `smart_money.py` | COT hedging pressure | De Roon et al. 2000, Journal of Finance; CFTC COT | experimental | no |
| `macro_context.py` / `ml_stock_lab/macro_features.py` | Macro context block | Fama-French conditional returns; Adrian-Shin 2010 | experimental | no |
| `regime_detection.py` | Market regime label | Ang-Bekaert 2002; Gen.is.IA internal rule system | implemented | no |
| `alpha101.py` | WorldQuant 101 alphas | Kakushadze 2016, arXiv:1601.00991 | experimental | no |
| `factors/eps_factors.py` | EPS surprise | Ball-Brown 1968; Bernard-Thomas 1989 | experimental | no |
| `factors/eps_factors.py` | EPS revision | Hawkins et al. 1984 analyst revision literature | experimental | no |
| `factors/eps_factors.py` | EPS forecast accuracy | Forecast-error evaluation / Gen.is.IA internal | experimental | no |
| `factors/eps_factors.py` | EPS growth momentum | Bernard-Thomas 1989, Journal of Accounting and Economics | experimental | no |
| `factors/eps_factors.py` | Earnings yield | Basu 1977, Journal of Finance | implemented / experimental source module | no |
| `simulation/monte_carlo.py` | Monte Carlo return paths | Glasserman 2003 | experimental | no |
| `simulation/monte_carlo.py` | Factor uncertainty bootstrap | Glasserman 2003; factor bootstrap validation practice | experimental | no |
| `colab/ml_training_pipeline.ipynb` | Colab Pro ML pipeline | Gen.is.IA cloud handoff methodology | implemented skeleton | no |

## High-Priority Gap Analysis

| Module | Factor | Reference | Status | Missing Doc? |
| --- | --- | --- | --- | --- |
| `factors/eps_factors.py` | `eps_surprise` | Ball-Brown 1968; Bernard-Thomas 1989 | implemented experimental | no |
| `factors/eps_factors.py` | `eps_revision` | Hawkins et al. 1984 | implemented experimental | no |
| `factors/eps_factors.py` | `eps_forecast_accuracy` | Gen.is.IA internal forecast evaluation | implemented experimental | no |
| `factors/eps_factors.py` | `eps_growth_momentum` | Bernard-Thomas 1989 | implemented experimental | no |
| `factors/eps_factors.py` | `earnings_yield` | Basu 1977 | implemented; metadata improved | no |
| `simulation/monte_carlo.py` | `monte_carlo_returns` | Glasserman 2003 | implemented experimental | no |
| `simulation/monte_carlo.py` | `monte_carlo_factor_uncertainty` | Glasserman 2003 | implemented experimental | no |
| `colab/ml_training_pipeline.ipynb` | Colab Pro ML pipeline | Gen.is.IA cloud handoff | implemented skeleton | no |
| `cross_asset_factors.py` | `cross_asset_momentum` | Asness-Moskowitz-Pedersen 2013 | implemented alias to `build_cross_asset_momentum` | no |
| `cross_asset_factors.py` | `cross_asset_value` | Asness-Moskowitz-Pedersen 2013 | implemented experimental | no |
| `cross_asset_factors.py` | `global_risk_factor` | PCA-based global risk-factor literature | implemented experimental | no |
| `cross_asset_factors.py` | `liquidity_factor` | Amihud 2002; Pastor-Stambaugh 2003 | STUB - owner: Data Platform; timeline: after Amihud/volume panel is standardized | yes |

## DATA_ARTIFACTS

| Artifact / manifest | State | CI impact | Note |
| --- | --- | --- | --- |
| `output/data_completion/*_2000_2026.csv` manifests | LOCAL_ONLY - not required for CI | none | `validate_research_data_coverage.py` exits 0 but can report these staging manifests as `MISSING` on a clean local checkout. Treat as artifact availability, not code regression. |
| Runtime training/log outputs under `output/logs/`, `output/ml_training_lab/`, `output/ml_stock_lab/` | LOCAL_ONLY - not required for CI | none | Heavy or generated outputs remain untracked by policy; sync lightweight summaries only when explicitly requested. |

## PR / Merge Checklist

| Item | Status | Note |
| --- | --- | --- |
| STUB factors documented with owner and timeline | done | `liquidity_factor` remains the only cross-asset STUB from this audit. |
| data_completion manifest gap added as DATA_ARTIFACTS note | done | Classified as `LOCAL_ONLY - not required for CI`. |
| CHANGELOG.md updated with v2.x entry | done | See root `CHANGELOG.md`. |
| PR description includes commits `1c849c7`, `223903f`, `901e7e4` | ready | Include these commit IDs in the PR body. |
| EPS lag reviewed for ticker edge cases | done | EPS factors use per-ticker fiscal-period shift and naive consensus proxy when unavailable. |
| Monte Carlo seed reproducibility reviewed | done | `seed=42` uses NumPy Generator semantics compatible with NumPy >= 1.17. |

## Residual Notes

- EPS consensus coverage is provider-dependent. The implementation uses a rolling historical EPS mean as a naive proxy when consensus is unavailable.
- Monte Carlo simulation assumes the supplied factor matrix is already point-in-time safe.
- The Colab notebook is a handoff skeleton, not a committed heavy training run.
- Cross-asset value and PCA global risk are implemented as experimental utilities. Cross-asset liquidity remains a STUB until the Amihud/volume panel is standardized across asset classes.
