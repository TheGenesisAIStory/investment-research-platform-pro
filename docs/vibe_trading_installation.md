# Vibe-Trading Installation

This repository vendors Vibe-Trading as a Git submodule at `vibe_trading/` and
keeps local glue code in `integrations/vibe_trading_bridge/`.

## Add Or Update The Submodule

Fresh checkout:

```bash
git submodule add https://github.com/HKUDS/Vibe-Trading.git vibe_trading
git submodule update --init --recursive
```

Existing checkout:

```bash
git submodule update --init --recursive
cd vibe_trading
git fetch origin
git checkout main
git pull --ff-only
cd ..
git add .gitmodules vibe_trading
```

## Python Environment

The bridge code is Python 3.10+, but Vibe-Trading currently declares
`requires-python >=3.11`. Use Python 3.11 or newer for the combined environment.

```bash
python3.11 -m venv .venv-vibe
source .venv-vibe/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-vibe.txt
```

If you prefer conda:

```bash
conda create -n ml4t-vibe python=3.11 -y
conda activate ml4t-vibe
python -m pip install --upgrade pip
python -m pip install -r requirements-vibe.txt
```

Keep heavier Vibe dependencies separate from the base project. The intended
split is:

- Base ML4T dependencies: whatever your notebooks and `ml_stock_lab` already
  use.
- Vibe extra: `requirements-vibe.txt`, which installs the submodule package and
  its agent dependencies.

## Configure Runtime Roots

```bash
export ML4T_REPO_ROOT="$(pwd)"
export VIBE_TRADING_ALLOWED_FILE_ROOTS="$(pwd),$(pwd)/output,$HOME/.vibe-trading/uploads"
export VIBE_TRADING_ALLOWED_RUN_ROOTS="$(pwd),$(pwd)/output,$HOME/.vibe-trading"
```

Initialize Vibe-Trading credentials or local provider settings:

```bash
cd vibe_trading
cp agent/.env.example agent/.env
# Edit agent/.env with your provider, model and optional data-source tokens.
cd ..
```

You can also run Vibe's guided setup:

```bash
source .venv-vibe/bin/activate
cd vibe_trading/agent
python -m cli init
```

## Run Vibe-Trading Within This Project

Interactive Vibe CLI:

```bash
source .venv-vibe/bin/activate
cd vibe_trading/agent
PYTHONPATH="$(pwd):$ML4T_REPO_ROOT" python -m cli
```

Single research run:

```bash
source .venv-vibe/bin/activate
python scripts/vibe_cli.py research \
  --prompt "Research quality and momentum signals for Italian banks" \
  --universe "ISP.MI,UCG.MI,BAMI.MI"
```

Generate a local strategy scaffold while also sending the enriched prompt to
Vibe when configured:

```bash
python scripts/vibe_cli.py generate-strategy \
  --prompt "12-month momentum with 1-month reversal filter and volatility cap" \
  --universe "AAPL,MSFT,NVDA" \
  --output-path strategies/generated_momentum.py
```

Backtest from a prompt:

```bash
python scripts/vibe_cli.py backtest-from-prompt \
  --prompt "mean reversion on EURUSD using z-scored 20-day returns" \
  --universe "EURUSD" \
  --start 2015-01-01 \
  --end 2024-12-31 \
  --transaction-cost-bps 1
```

For dry runs that do not call the Vibe CLI, add `--no-vibe`.

