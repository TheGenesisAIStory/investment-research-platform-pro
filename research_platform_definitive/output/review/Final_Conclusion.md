# Final Conclusion: TCEL vs IFRS 9 ECL

## Key empirical findings
- EL_ECL is more volatile than EL_TCEL: the mean portfolio volatility ratio is **5.25x**.
- EL_ECL is more macro-sensitive: its correlation with macro stress is **0.53**, versus **-0.00** for EL_TCEL.
- TCEL-based EVA/RAROC are more stable: the mean EVA volatility ratio ECL/TCEL is **1.29x**.
- Macro and staging variables explain a substantial share of the ECL-TCEL gap in the panel regressions.
- Walk-forward ML produces measurable predictive signal for future TCEL EVA; best average RMSE is **78.3944**.
- Placebo performance weakens materially: best placebo RMSE is **146.5308**.
- The leading predictive drivers are: ead_lag1_w, el_tcel_lag1_w, raroc_tcel_lag1_w, economic_capital_lag1_w, eva_tcel_lag1_w.

## Methodological caveats
The current package is publication-ready as a reproducible methodological notebook. If confidential bank data are unavailable, the synthetic panel should be described as a controlled numerical experiment rather than external empirical proof.

## Practical implications
TCEL is a stable structural loss metric for EVA, RAROC and economic-capital steering. IFRS 9 ECL remains valuable for accounting and forward-looking risk recognition, but its macro and staging sensitivity makes it less suitable as the sole internal management metric.

## Suggested next research steps
1. Replace synthetic data with audited bank portfolio-quarter observations.
2. Estimate clustered or hierarchical panel models by portfolio and macro regime.
3. Extend the ML layer with explainable PD/LGD component models.
4. Reconcile TCEL with regulatory EL, pricing spreads and capital allocation rules.
