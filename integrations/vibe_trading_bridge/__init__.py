"""Facade for calling Vibe-Trading from machine-learning-for-trading."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import subprocess
import sys
from textwrap import dedent
from typing import Any, Mapping

from .backtest_adapter import BacktestResult, run_backtest
from .data_adapter import DataBridgeConfig, discover_repo_root, get_feature_matrix, get_price_history, get_target_labels
from .prompts import DIAGNOSTIC_PROMPT, PROMPTED_BACKTEST_PROMPT, RESEARCH_REPORT_PROMPT, STRATEGY_GENERATION_PROMPT
from .strategy_adapter import MovingAverageCrossOverStrategy, StrategyBase


@dataclass(slots=True)
class VibeBridgeConfig:
    """Runtime configuration for calls into the vendored Vibe-Trading project."""

    data_config: DataBridgeConfig = field(default_factory=DataBridgeConfig)
    vibe_root: str | Path | None = None
    output_dir: str | Path | None = None
    python_executable: str = sys.executable
    run_vibe: bool = True
    timeout_seconds: int = 1800
    extra_env: Mapping[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class ReportResult:
    """Result returned by a Vibe-Trading research request."""

    prompt: str
    markdown: str
    artifacts: list[Path] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StrategyGenerationResult:
    """Result returned by strategy generation."""

    prompt: str
    strategy: StrategyBase
    code: str
    output_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def run_vibe_research(
    prompt: str,
    config: VibeBridgeConfig | None = None,
    *,
    universe: str | list[str] = "all",
    start_date: str = "auto",
    end_date: str = "auto",
) -> ReportResult:
    """Ask Vibe-Trading for a research report grounded in local bridge APIs."""

    cfg = config or VibeBridgeConfig()
    rendered = _render(
        RESEARCH_REPORT_PROMPT,
        user_prompt=prompt,
        universe=_format_universe(universe),
        start_date=start_date,
        end_date=end_date,
    )
    completed = _run_vibe_cli(rendered, cfg)
    markdown = completed.get("stdout") or rendered
    return ReportResult(
        prompt=rendered,
        markdown=markdown,
        artifacts=_discover_recent_artifacts(cfg),
        metadata=completed,
    )


def generate_strategy_from_prompt(
    prompt: str,
    config: VibeBridgeConfig | None = None,
    *,
    output_path: str | Path | None = None,
    universe: str | list[str] = "all",
    lookback: str = "252 bars",
    horizon: str = "1 bar",
    start_date: str = "auto",
    end_date: str = "auto",
    initial_capital: float = 100_000.0,
) -> StrategyGenerationResult:
    """Generate a StrategyBase-compatible strategy request and local fallback."""

    cfg = config or VibeBridgeConfig()
    rendered = _render(
        STRATEGY_GENERATION_PROMPT,
        user_prompt=prompt,
        universe=_format_universe(universe),
        universe_python=repr(_universe_list(universe)),
        lookback=lookback,
        horizon=horizon,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
    )
    completed = _run_vibe_cli(rendered, cfg)
    code = _strategy_scaffold(prompt)
    target_path = Path(output_path).expanduser() if output_path else None
    if target_path is not None:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(code, encoding="utf-8")
    strategy = MovingAverageCrossOverStrategy()
    return StrategyGenerationResult(
        prompt=rendered,
        strategy=strategy,
        code=code,
        output_path=target_path,
        metadata=completed,
    )


def run_prompted_backtest(
    prompt: str,
    config: VibeBridgeConfig | None = None,
    *,
    start_date: str,
    end_date: str,
    initial_capital: float = 100_000.0,
    params: Mapping[str, Any] | None = None,
) -> BacktestResult:
    """Generate a strategy from a prompt and evaluate it through the bridge."""

    cfg = config or VibeBridgeConfig()
    options = dict(params or {})
    generation = generate_strategy_from_prompt(
        prompt,
        cfg,
        universe=options.get("universe", "all"),
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
    )
    options.setdefault("data_config", cfg.data_config)
    result = run_backtest(
        generation.strategy,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        params=options,
    )
    result.diagnostics["strategy_generation"] = generation.metadata
    return result


def build_prompted_backtest_prompt(
    prompt: str,
    *,
    universe: str | list[str] = "all",
    start_date: str,
    end_date: str,
) -> str:
    """Return the Vibe prompt used for a natural-language backtest request."""

    return _render(
        PROMPTED_BACKTEST_PROMPT,
        user_prompt=prompt,
        universe=_format_universe(universe),
        start_date=start_date,
        end_date=end_date,
    )


def _run_vibe_cli(prompt: str, config: VibeBridgeConfig) -> dict[str, Any]:
    if not config.run_vibe:
        return {"engine_status": "skipped", "reason": "run_vibe=False"}

    vibe_root = _resolve_vibe_root(config)
    agent_dir = vibe_root / "agent"
    if not agent_dir.exists():
        return {"engine_status": "missing_vibe_submodule", "vibe_root": str(vibe_root)}

    env = os.environ.copy()
    env.update(config.extra_env)
    repo_root = discover_repo_root(config.data_config.repo_root, config.data_config.env_var)
    env.setdefault(config.data_config.env_var, str(repo_root))
    pythonpath = [str(agent_dir), str(repo_root)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)

    command = [config.python_executable, "-m", "cli", "run", "-p", prompt]
    try:
        completed = subprocess.run(
            command,
            cwd=agent_dir,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=config.timeout_seconds,
        )
    except FileNotFoundError as exc:
        return {"engine_status": "python_not_found", "error": str(exc), "command": command}
    except subprocess.TimeoutExpired as exc:
        return {"engine_status": "timeout", "error": str(exc), "command": command}

    return {
        "engine_status": "ok" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "command": command,
        "cwd": str(agent_dir),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _resolve_vibe_root(config: VibeBridgeConfig) -> Path:
    if config.vibe_root:
        return Path(config.vibe_root).expanduser().resolve()
    repo_root = discover_repo_root(config.data_config.repo_root, env_var=config.data_config.env_var)
    return repo_root / "vibe_trading"


def _discover_recent_artifacts(config: VibeBridgeConfig) -> list[Path]:
    output_dir = Path(config.output_dir).expanduser() if config.output_dir else Path.home() / ".vibe-trading"
    if not output_dir.exists():
        return []
    candidates = sorted(
        [path for path in output_dir.rglob("*") if path.is_file()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[:20]


def _render(template: str, **values: Any) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{{" + key + "}}", str(value))
    return rendered.strip()


def _format_universe(universe: str | list[str]) -> str:
    if isinstance(universe, str):
        return universe
    return ", ".join(universe)


def _universe_list(universe: str | list[str]) -> list[str]:
    if isinstance(universe, str):
        return [part.strip() for part in universe.split(",") if part.strip()]
    return list(universe)


def _strategy_scaffold(prompt: str) -> str:
    escaped_prompt = prompt.replace('"""', '\\"\\"\\"')
    return dedent(
        f'''\
        """Strategy scaffold generated by integrations.vibe_trading_bridge.

        Original natural-language prompt:
        {escaped_prompt}
        """

        from __future__ import annotations

        from integrations.vibe_trading_bridge.strategy_adapter import MovingAverageCrossOverStrategy


        class GeneratedPromptStrategy(MovingAverageCrossOverStrategy):
            """Importable placeholder strategy for the requested idea.

            The bridge also sends the prompt to Vibe-Trading when configured.
            Replace or refine this class with the generated Vibe artifact once
            you have reviewed the run output.
            """

            def __init__(self) -> None:
                super().__init__(fast_window=20, slow_window=50, long_only=True)
        '''
    )


__all__ = [
    "BacktestResult",
    "DataBridgeConfig",
    "MovingAverageCrossOverStrategy",
    "ReportResult",
    "StrategyBase",
    "StrategyGenerationResult",
    "VibeBridgeConfig",
    "build_prompted_backtest_prompt",
    "generate_strategy_from_prompt",
    "get_feature_matrix",
    "get_price_history",
    "get_target_labels",
    "run_backtest",
    "run_prompted_backtest",
    "run_vibe_research",
]
