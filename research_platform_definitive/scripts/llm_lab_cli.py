#!/usr/bin/env python3
"""Render and inspect Italian LLM Lab prompt packets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from research_platform_core.llm_lab import (  # noqa: E402
    available_templates,
    build_context_from_database,
    provider_registry_frame,
    render_prompt,
    save_prompt_packet,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLM Lab prompt renderer for vibe coding and vibe trading.")
    parser.add_argument("--db-path", default=None, help="Optional SQLite research DB path.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("templates", help="List available Italian prompt templates.")
    subparsers.add_parser("providers", help="Show provider readiness from environment variables.")

    render = subparsers.add_parser("render", help="Render one template to stdout.")
    render.add_argument("template_name")

    packet = subparsers.add_parser("packet", help="Render all prompt files to output/llm_lab.")
    packet.add_argument("--output-dir", default=None)
    packet.add_argument("--template", action="append", default=None, help="Template to include; can be repeated.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "templates":
        print(available_templates().to_csv(index=False))
        return 0
    if args.command == "providers":
        print(provider_registry_frame().to_csv(index=False))
        return 0
    if args.command == "render":
        context = build_context_from_database(args.db_path)
        print(render_prompt(args.template_name, context))
        return 0
    if args.command == "packet":
        paths = save_prompt_packet(args.template, output_dir=args.output_dir, db_path=args.db_path)
        print("llm_lab_prompt_packet_OK")
        print(json.dumps(paths, indent=2, sort_keys=True))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
