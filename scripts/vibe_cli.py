#!/usr/bin/env python
"""Command-line entrypoints for the Vibe-Trading bridge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from integrations.vibe_trading_bridge import (  # noqa: E402
    DataBridgeConfig,
    VibeBridgeConfig,
    generate_strategy_from_prompt,
    run_prompted_backtest,
    run_vibe_research,
)


def main(argv: list[str] | None = None) -> int:
    """Run the bridge CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)
    config = _build_config(args)

    if args.command == "research":
        result = run_vibe_research(
            args.prompt,
            config,
            universe=args.universe,
            start_date=args.start or "auto",
            end_date=args.end or "auto",
        )
        if args.output_path:
            path = Path(args.output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(result.markdown, encoding="utf-8")
        print(result.markdown)
        print(json.dumps({"metadata": _jsonable(result.metadata), "artifacts": [str(path) for path in result.artifacts]}, indent=2))
        return 0

    if args.command == "generate-strategy":
        result = generate_strategy_from_prompt(
            args.prompt,
            config,
            output_path=args.output_path,
            universe=args.universe,
            lookback=args.lookback,
            horizon=args.horizon,
        )
        if result.output_path:
            print(f"Wrote strategy scaffold: {result.output_path}")
        print(json.dumps({"metadata": _jsonable(result.metadata)}, indent=2))
        return 0

    if args.command == "backtest-from-prompt":
        params: dict[str, Any] = {
            "universe": _parse_universe(args.universe),
            "transaction_cost_bps": args.transaction_cost_bps,
            "allow_short": args.allow_short,
            "max_gross_exposure": args.max_gross_exposure,
        }
        result = run_prompted_backtest(
            args.prompt,
            config,
            start_date=args.start,
            end_date=args.end,
            initial_capital=args.initial_capital,
            params=params,
        )
        print(json.dumps(result.to_dict(), indent=2))
        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            result.returns_curve.to_csv(output_dir / "returns_curve.csv", index=False)
            result.trades.to_csv(output_dir / "trades.csv", index=False)
            result.weights.to_csv(output_dir / "weights.csv", index=False)
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser for bridge commands."""

    parser = argparse.ArgumentParser(description="Vibe-Trading bridge for machine-learning-for-trading")
    parser.add_argument("--repo-root", default=None, help="Project root; defaults to ML4T_REPO_ROOT or parent discovery.")
    parser.add_argument("--output-root", default=None, help="Optional project output root used by data loaders.")
    parser.add_argument("--financial-db-root", default=None, help="Optional local financial database root.")
    parser.add_argument("--vibe-root", default=None, help="Path to vendored Vibe-Trading checkout.")
    parser.add_argument("--no-vibe", action="store_true", help="Skip calling the Vibe-Trading CLI and use local bridge fallbacks.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    research = subparsers.add_parser("research", help="Generate a research report from a natural-language prompt.")
    research.add_argument("--prompt", required=True)
    research.add_argument("--universe", default="all")
    research.add_argument("--start", default=None)
    research.add_argument("--end", default=None)
    research.add_argument("--output-path", default=None)

    generate = subparsers.add_parser("generate-strategy", help="Generate a StrategyBase-compatible strategy scaffold.")
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--output-path", required=True)
    generate.add_argument("--universe", default="all")
    generate.add_argument("--lookback", default="252 bars")
    generate.add_argument("--horizon", default="1 bar")

    backtest = subparsers.add_parser("backtest-from-prompt", help="Generate and backtest a strategy from a prompt.")
    backtest.add_argument("--prompt", required=True)
    backtest.add_argument("--start", required=True)
    backtest.add_argument("--end", required=True)
    backtest.add_argument("--universe", default="all")
    backtest.add_argument("--initial-capital", type=float, default=100_000.0)
    backtest.add_argument("--transaction-cost-bps", type=float, default=0.0)
    backtest.add_argument("--max-gross-exposure", type=float, default=1.0)
    backtest.add_argument("--allow-short", action="store_true")
    backtest.add_argument("--output-dir", default=None)

    return parser


def _build_config(args: argparse.Namespace) -> VibeBridgeConfig:
    data_config = DataBridgeConfig(
        repo_root=args.repo_root,
        output_root=args.output_root,
        financial_db_root=args.financial_db_root,
    )
    return VibeBridgeConfig(data_config=data_config, vibe_root=args.vibe_root, run_vibe=not args.no_vibe)


def _parse_universe(value: str) -> list[str] | str:
    if value.lower() in {"all", "sp500", "s&p 500", "csi300"}:
        return value
    return [part.strip().upper() for part in value.split(",") if part.strip()]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


if __name__ == "__main__":
    raise SystemExit(main())

