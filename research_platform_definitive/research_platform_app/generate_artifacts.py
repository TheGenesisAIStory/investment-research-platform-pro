"""CLI helper to regenerate lightweight Streamlit-facing artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from operations import generate_all_artifacts, generate_company_artifacts, generate_portfolio_artifacts
from support import default_roots


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate lightweight research-platform artifacts.")
    parser.add_argument("--domain", choices=["all", "company", "portfolio"], default="all")
    parser.add_argument("--company-root", type=Path, default=None)
    parser.add_argument("--portfolio-root", type=Path, default=None)
    parser.add_argument("--workspace-root", type=Path, default=None)
    args = parser.parse_args()

    roots = default_roots()
    if args.company_root is not None:
        roots["company"] = args.company_root.expanduser()
    if args.portfolio_root is not None:
        roots["portfolio"] = args.portfolio_root.expanduser()
    if args.workspace_root is not None:
        roots["workspace"] = args.workspace_root.expanduser()

    if args.domain == "company":
        result = {"company": generate_company_artifacts(roots["company"])}
    elif args.domain == "portfolio":
        result = {"portfolio": generate_portfolio_artifacts(roots["portfolio"], roots["workspace"])}
    else:
        result = generate_all_artifacts(roots)

    print(json.dumps(result, indent=2, default=str))
    return 0 if all(item.get("ok") for item in result.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
