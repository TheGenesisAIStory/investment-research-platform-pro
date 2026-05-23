"""Prompt templates for Vibe-Trading research and strategy generation."""

RESEARCH_REPORT_PROMPT = """
You are working inside the machine-learning-for-trading repository.

User research request:
{{user_prompt}}

Universe:
{{universe}}

Date range:
{{start_date}} to {{end_date}}

Use the local bridge APIs instead of downloading duplicate data:
- integrations.vibe_trading_bridge.data_adapter.get_price_history
- integrations.vibe_trading_bridge.data_adapter.get_feature_matrix
- integrations.vibe_trading_bridge.data_adapter.get_target_labels

Produce:
1. A Markdown research report with thesis, data coverage, feature diagnostics,
   signal intuition, risks, and next experiments.
2. A runnable Python notebook or script that imports the bridge APIs and can be
   rerun against the same universe.
3. A compact artifact manifest listing generated charts, tables, and code.

Constraints:
- Do not assume live-trading execution.
- Do not hard-code absolute local paths.
- Prefer point-in-time-safe features and explicitly flag any possible lookahead
  risk.
- Use existing project metrics for Sharpe, drawdown, turnover, and validation
  whenever possible.
"""


STRATEGY_GENERATION_PROMPT = """
You are generating a strategy for machine-learning-for-trading.

User strategy idea:
{{user_prompt}}

Target universe:
{{universe}}

Lookback / horizon:
{{lookback}} / {{horizon}}

Write Python code that plugs into this interface:

from integrations.vibe_trading_bridge.strategy_adapter import StrategyBase, StrategyContext

class GeneratedStrategy(StrategyBase):
    \"\"\"Explain the full signal formula, parameters, filters, rebalancing rule,
    risk management, and expected failure modes here.\"\"\"

    def on_bar(self, context: StrategyContext, historical_data, features=None):
        ...

The generated class must:
- Inherit from StrategyBase.
- Return a mapping of ticker to signal or a DataFrame with date, ticker and
  target_weight/signal.
- Use only data passed through historical_data and features.
- Avoid lookahead by using data available at or before context.date.
- Expose clear __init__ parameters with type hints.
- Include concise comments for non-obvious formulas.
- Be compatible with integrations.vibe_trading_bridge.backtest_adapter.run_backtest.

Also include a short test snippet:

from integrations.vibe_trading_bridge.backtest_adapter import run_backtest
result = run_backtest(
    GeneratedStrategy(...),
    start_date="{{start_date}}",
    end_date="{{end_date}}",
    initial_capital={{initial_capital}},
    params={"universe": {{universe_python}}},
)
print(result.metrics)
"""


DIAGNOSTIC_PROMPT = """
You are diagnosing a backtest from machine-learning-for-trading.

Original idea:
{{user_prompt}}

Backtest metrics:
{{metrics}}

Diagnostics and logs:
{{diagnostics}}

Analyze:
1. Whether performance is likely driven by robust signal quality, leverage,
   costs, data leakage, regime concentration, or a small number of trades.
2. Which metrics are missing for a credible research decision.
3. Specific modifications to try next, including parameter sweeps, data
   filters, risk controls, and ablation tests.
4. A prioritized experiment queue that can be run through
   integrations.vibe_trading_bridge.backtest_adapter.run_backtest.

Output a concise Markdown diagnostic memo and runnable Python snippets for the
top experiments. Do not recommend live trading.
"""


PROMPTED_BACKTEST_PROMPT = """
Turn the following natural-language idea into a StrategyBase-compatible class,
then run it through the local bridge backtest API.

Idea:
{{user_prompt}}

Backtest window:
{{start_date}} to {{end_date}}

Universe:
{{universe}}

Use:
- get_price_history for local prices.
- get_feature_matrix for local features.
- run_backtest for evaluation.

Return the generated strategy code, metrics, assumptions, and an explanation of
any fallbacks or missing data.
"""

