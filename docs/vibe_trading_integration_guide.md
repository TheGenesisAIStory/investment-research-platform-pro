# Vibe-Trading Integration Guide

## Architecture

```
Natural-language prompt
        |
        v
scripts/vibe_cli.py or Python API
        |
        v
integrations/vibe_trading_bridge
  |-- prompts.py              prompt contracts for Vibe
  |-- data_adapter.py         local prices, features, labels
  |-- strategy_adapter.py     StrategyBase interface
  |-- backtest_adapter.py     standardized BacktestResult
        |
        +--> ml_stock_lab / research_platform_definitive
        |
        +--> vibe_trading/    vendored HKUDS/Vibe-Trading submodule
```

Reverse direction:

```
Vibe-Trading generated code
        |
        v
from integrations.vibe_trading_bridge import ...
        |
        +--> get_price_history(...)
        +--> get_feature_matrix(...)
        +--> run_backtest(...)
        +--> project metrics: Sharpe, drawdown, turnover
```

## Repository Layout

```text
machine-learning-for-trading/
  ml_stock_lab/
  research_platform_definitive/
  scripts/
    vibe_cli.py
  vibe_trading/
    agent/
    frontend/
    pyproject.toml
  integrations/
    __init__.py
    vibe_trading_bridge/
      __init__.py
      prompts.py
      data_adapter.py
      strategy_adapter.py
      backtest_adapter.py
  docs/
    vibe_trading_installation.md
    vibe_trading_integration_guide.md
  requirements-vibe.txt
```

## Workflow 1: Prompt To Strategy To Backtest

1. User submits an idea:

```bash
python scripts/vibe_cli.py backtest-from-prompt \
  --prompt "20/50 moving-average trend strategy with monthly rebalance" \
  --universe "AAPL,MSFT,NVDA" \
  --start 2018-01-01 \
  --end 2024-12-31 \
  --output-dir output/vibe_runs/ma_example
```

2. The bridge wraps the idea with `STRATEGY_GENERATION_PROMPT`, asking Vibe to
   generate a class inheriting from `StrategyBase`.

3. The backtest adapter loads local prices/features with:

```python
from integrations.vibe_trading_bridge import DataBridgeConfig, run_backtest
from integrations.vibe_trading_bridge.strategy_adapter import MovingAverageCrossOverStrategy

config = DataBridgeConfig(repo_root=".")
strategy = MovingAverageCrossOverStrategy(fast_window=20, slow_window=50)
result = run_backtest(
    strategy,
    start_date="2018-01-01",
    end_date="2024-12-31",
    initial_capital=100_000,
    params={"universe": ["AAPL", "MSFT", "NVDA"], "data_config": config},
)
print(result.metrics)
```

4. Output is standardized as `BacktestResult`:

- `returns_curve`: date, return, gross return, transaction cost, equity.
- `trades`: date, ticker, delta weight, target weight, price.
- `weights`: long-form target weights.
- `metrics`: total return, annualized return, volatility, Sharpe, max drawdown,
  turnover and observation count.

## Workflow 2: Prompt To Research Report

```bash
python scripts/vibe_cli.py research \
  --prompt "Compare quality, value and momentum signals for Italian banks" \
  --universe "ISP.MI,UCG.MI,BAMI.MI" \
  --start 2015-01-01 \
  --end 2024-12-31 \
  --output-path output/vibe_runs/italian_banks_report.md
```

The research prompt instructs Vibe to use:

```python
from integrations.vibe_trading_bridge.data_adapter import (
    get_feature_matrix,
    get_price_history,
    get_target_labels,
)
```

This keeps generated notebooks and reports on the same local data layer as the
rest of the project.

## Complete Python Example

```python
from integrations.vibe_trading_bridge import (
    DataBridgeConfig,
    VibeBridgeConfig,
    generate_strategy_from_prompt,
    run_prompted_backtest,
    run_vibe_research,
)

data_config = DataBridgeConfig(repo_root=".", output_root="output")
vibe_config = VibeBridgeConfig(data_config=data_config, run_vibe=True)

report = run_vibe_research(
    "Find robust quality and momentum combinations for my bank universe.",
    vibe_config,
    universe=["ISP.MI", "UCG.MI", "BAMI.MI"],
    start_date="2015-01-01",
    end_date="2024-12-31",
)

generated = generate_strategy_from_prompt(
    "Long top quality-value names, avoid high drawdown stocks.",
    vibe_config,
    output_path="strategies/quality_value_generated.py",
    universe=["ISP.MI", "UCG.MI", "BAMI.MI"],
)

result = run_prompted_backtest(
    "Long top quality-value names, avoid high drawdown stocks.",
    vibe_config,
    start_date="2015-01-01",
    end_date="2024-12-31",
    params={"universe": ["ISP.MI", "UCG.MI", "BAMI.MI"], "transaction_cost_bps": 5},
)

print(result.metrics)
```

## Extending The Bridge

For tighter production integration, inject callables instead of relying on the
default artifact scanner:

```python
from integrations.vibe_trading_bridge import DataBridgeConfig

def load_prices(symbols, start_date, end_date, fields, config):
    # Call your preferred data store here.
    ...

data_config = DataBridgeConfig(price_loader=load_prices)
```

Use the same pattern for `feature_loader`, `target_loader`, or
`params["backtest_runner"]` when you want the bridge to call a specific
portfolio simulator.

