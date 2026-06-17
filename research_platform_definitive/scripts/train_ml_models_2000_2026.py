#!/usr/bin/env python3
"""Train ML Stock Lab expected-return models on the 2000-2026 factor panel."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

from ml_stock_lab.training import train_ml_model_suite
from research_platform_core.run_lock import acquire_stage_lock, read_stage_lock, release_stage_lock


def _models(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _copy_alpha101_artifacts(paths: dict[str, str]) -> dict[str, str]:
    copied: dict[str, str] = {}
    for key, value in paths.items():
        path = Path(str(value))
        if not path.exists() or not path.is_file():
            continue
        target = path.with_name(f"{path.stem}_alpha101{path.suffix}")
        try:
            shutil.copy2(path, target)
            copied[f"{key}_alpha101"] = str(target)
        except OSError:
            continue
    return copied


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--train-end-year", type=int, default=2018)
    parser.add_argument("--test-start-year", type=int, default=2019)
    parser.add_argument("--models", default="ols,rf", help="Comma-separated model families: ols,lasso,rf,gbrt,ensemble.")
    parser.add_argument("--feature-blocks", default="value,quality,momentum,risk,size,growth,model_based", help="Comma-separated factor blocks for model features.")
    parser.add_argument("--target-horizon-days", type=int, default=21, help="Forward return horizon used as the expected-return target.")
    parser.add_argument("--cost-bps", type=float, default=10.0, help="Round-trip transaction-cost assumption for net long-short diagnostics.")
    parser.add_argument("--max-rows", type=int, default=0, help="Cap rows for smoke/debug. Use 0 for full panel.")
    parser.add_argument("--financial-db-root", default=None)
    parser.add_argument("--output-root", default=str(ROOT / "output"))
    parser.add_argument("--use-alpha101", nargs="?", const="true", default="false", help="Append alpha101 to feature blocks and write _alpha101 artifact copies.")
    parser.add_argument("--use-ollama", action="store_true", help="Ask local Ollama for an Italian research summary if available.")
    parser.add_argument("--ollama-model", default="llama3.1")
    args = parser.parse_args()

    output_root = Path(args.output_root).expanduser()
    feature_blocks = _models(args.feature_blocks)
    use_alpha101 = _truthy(args.use_alpha101)
    if use_alpha101 and "alpha101" not in feature_blocks:
        feature_blocks.append("alpha101")
    lock = acquire_stage_lock("ml_training", output_root, protected_root=output_root / "ml_training_lab")
    if lock is None:
        existing = read_stage_lock("ml_training", output_root)
        print("ml_training_locked_running")
        print(f"locked_by={getattr(existing, 'run_id', '')}")
        print(f"pid={getattr(existing, 'pid', '')}")
        print(f"lock_path={getattr(existing, 'lock_path', '')}")
        return 2
    try:
        result = train_ml_model_suite(
            output_root=args.output_root,
            financial_db_root=args.financial_db_root,
            start_year=args.start_year,
            end_year=args.end_year,
            train_end_year=args.train_end_year,
            test_start_year=args.test_start_year,
            models=_models(args.models),
            max_rows=None if args.max_rows == 0 else args.max_rows,
            use_ollama=args.use_ollama,
            ollama_model=args.ollama_model,
            target_horizon_days=args.target_horizon_days,
            feature_blocks=feature_blocks,
            cost_bps=args.cost_bps,
        )
        if use_alpha101:
            result["paths"].update(_copy_alpha101_artifacts(result.get("paths", {})))
    finally:
        release_stage_lock("ml_training", output_root, lock_info=lock)
    print("ml_training_complete")
    print(f"status={result['status']}")
    print(f"metrics_rows={len(result['metrics'])}")
    print(f"prediction_rows={len(result['predictions'])}")
    for key, path in result["paths"].items():
        print(f"{key}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
